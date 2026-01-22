import json
import time
import re
import string
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple, Optional, Any

from tqdm import tqdm
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "behave_full_content.json"
OUTPUT_FILE = "behave_propositions.json"
MODEL_NAME = "qwen2.5:3b"  # Hoặc "qwen2.5:7b"
TEST_MODE = True  # True: Chỉ chạy thử 3 chunk đầu tiên để test. False: Chạy cả sách.

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z“"\'])')


# ============================================================
# DEDUPE / NOISE FILTER
# ============================================================

def _norm_for_dedupe(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    return s


def _looks_like_noise(s: str) -> bool:
    s = s.strip()

    if not s:
        return True

    # chỉ là số
    if re.fullmatch(r'\d+', s):
        return True

    # câu hỏi (KG thường không lấy)
    if s.endswith('?'):
        return True

    # mảnh câu: kết thúc bằng dấu phẩy hoặc quá ngắn
    if s.endswith(',') or len(s) < 18:
        return True

    # mệnh đề phụ hay gây rác
    if re.match(r'^(and|but|which|when|while|because|as will be|instead)\b', s.lower()):
        return True

    return False


def sanitize_props(props):
    cleaned = []
    for p in props:
        p = re.sub(r'\s+', ' ', p).strip()
        if _looks_like_noise(p):
            continue
        cleaned.append(p)

    # dedupe fuzzy
    kept = []
    kept_norm = []
    for p in cleaned:
        np = _norm_for_dedupe(p)
        dup = False
        for n2 in kept_norm:
            if SequenceMatcher(a=np, b=n2).ratio() >= 0.92:
                dup = True
                break
        if not dup:
            kept.append(p)
            kept_norm.append(np)

    return kept


# ============================================================
# TEXT NORMALIZATION + SENTENCE GROUPING
# ============================================================

def normalize_pdf_text(t: str) -> str:
    # 1) bỏ số trang đứng một mình (vd: "24")
    t = re.sub(r'(?m)^\s*\d+\s*$', '', t)

    # 2) nối từ bị ngắt bằng "-\n"  (vd: "third-\nparty" -> "thirdparty")
    t = re.sub(r'(\w)-\n(\w)', r'\1\2', t)

    # 3) đổi newline đơn trong câu thành khoảng trắng (giữ đoạn cách bằng 2 newline)
    t = re.sub(r'(?<!\n)\n(?!\n)', ' ', t)

    # 4) gom khoảng trắng thừa
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)

    return t.strip()


def split_into_sentences(text: str):
    text = normalize_pdf_text(text)
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def group_sentences(sentences, max_chars=1200):
    blocks = []
    cur = []
    cur_len = 0

    for s in sentences:
        if cur and cur_len + len(s) + 1 > max_chars:
            blocks.append(" ".join(cur))
            cur = []
            cur_len = 0
        cur.append(s)
        cur_len += len(s) + 1

    if cur:
        blocks.append(" ".join(cur))

    return blocks


def split_text_into_windows(text):
    """
    Cắt text thành các đoạn nhỏ vừa phải để LLM xử lý (khoảng 3-5 câu).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Đủ lớn để giữ ngữ cảnh 1 đoạn văn
        chunk_overlap=150,    # Overlap để không bị đứt mạch ý ở biên
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


# ============================================================
# LLM OUTPUT PARSING + DEBUG DUMP
# ============================================================

def strip_code_fences(s: str) -> str:
    # loại ```json ... ``` hoặc ``` ... ```
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def coerce_to_list_of_strings(data):
    # case 1: đúng format mong muốn
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return data

    # case 2: model trả dict bọc ngoài
    if isinstance(data, dict):
        preferred_keys = ["propositions", "statements", "items", "output", "result", "data"]
        for k in preferred_keys:
            v = data.get(k)
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return v
            if isinstance(v, list) and all(isinstance(x, dict) and "text" in x for x in v):
                return [x["text"] for x in v if isinstance(x.get("text"), str)]

        # fallback: tìm đại “list[str]” đầu tiên trong dict
        for v in data.values():
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return v

    return None


def parse_json_to_props(raw: str):
    cleaned = strip_code_fences(raw)
    data = json.loads(cleaned)
    props = coerce_to_list_of_strings(data)
    if props is None:
        return (
            False,
            None,
            f"JSON parsed but cannot coerce to list[str]. top_type={type(data)} "
            f"keys={list(data.keys()) if isinstance(data, dict) else None}",
        )
    return True, props, ""


def dump_debug(
    debug_dir: str,
    chapter_title: str,
    window_idx: int,
    window_text: str,
    llm_raw: str,
    parse_err: str,
):
    out = {
        "chapter": chapter_title,
        "window_idx": window_idx,
        "window_len": len(window_text),
        "window_head": window_text[:600],
        "window_tail": window_text[-600:],
        "llm_raw": llm_raw,
        "parse_error": parse_err,
    }

    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    Path(debug_dir, f"debug_{window_idx:04d}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# LLM CHAINS
# ============================================================

def setup_llm_chain():
    """
    Thiết lập LangChain với Qwen model và Prompt chuyên dụng.
    """
    llm = ChatOllama(model=MODEL_NAME, temperature=0, format="json")

    # Prompt kỹ thuật để tách ý và xử lý đại từ
    system_prompt = """
    You are an expert Knowledge Graph data pre-processor.
    Your task is to decompose the given text into "Atomic Facts" (short, standalone sentences).

    STRICT RULES:
    1. Split compound sentences into simple sentences.
    2. RESOLVE COREFERENCES: Replace pronouns (it, he, she, they, this, that) with the specific entities they refer to.
    3. Maintain original scientific terminology.
    4. Output MUST be a valid JSON list of strings.
    5. Do not add any explanation.

    Example Input: "The amygdala receives input from the cortex, but it relies on the PFC for regulation."
    Example Output: ["The amygdala receives input from the cortex.", "The amygdala relies on the PFC for regulation."]
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}"),
        ]
    )

    # Chain: Prompt -> LLM -> JSON Parser
    return prompt | llm | JsonOutputParser()


def setup_llm_chain_raw():
    system_prompt = """
    You are a helpful assistant that extracts atomic propositions.
    Rules:
    - Split complex sentences into simple, standalone statements.
    - Resolve pronouns when possible.
    - Preserve scientific terms.
    Output:
    - Return ONLY a JSON array of strings.
    - No markdown. No code fences. No extra text.
    """.strip()

    llm = ChatOllama(model=MODEL_NAME, temperature=0, format="json")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "TEXT:\n{text}\n\nReturn ONLY a JSON array of strings."),
        ]
    )

    return prompt | llm  # <-- raw AIMessage


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_chapter(chapter: dict, chain_raw, debug_dir: str = "debug_windows"):
    title = chapter.get("title", "Unknown")
    content = normalize_pdf_text(chapter.get("content", "") or "")
    sentences = split_into_sentences(content)
    windows = group_sentences(sentences, max_chars=1200)

    propositions = []

    for i, window in enumerate(windows):
        try:
            msg = chain_raw.invoke({"text": window})
            llm_raw = getattr(msg, "content", str(msg))

            ok, props, err = parse_json_to_props(llm_raw)
            if ok and props is not None:
                props = sanitize_props(props)

                for p in props:
                    propositions.append(
                        {
                            "text": p,
                            "source_chapter": title,
                            "original_window_index": i,
                        }
                    )
            else:
                dump_debug(debug_dir, title, i, window, llm_raw, err)
                propositions.append(
                    {
                        "text": window,
                        "source_chapter": title,
                        "note": f"raw_fallback: {err}",
                    }
                )

        except Exception as e:
            dump_debug(debug_dir, title, i, window, "", repr(e))
            propositions.append(
                {
                    "text": window,
                    "source_chapter": title,
                    "note": f"raw_fallback_exception: {repr(e)}",
                }
            )

    return propositions


def main():
    print(f"--- BẮT ĐẦU PROPOSITIONAL CHUNKING (Model: {MODEL_NAME}) ---")

    # 1. Setup
    try:
        # chain = setup_llm_chain()
        chain_raw = setup_llm_chain_raw()
    except Exception as e:
        print(f"❌ Lỗi kết nối Ollama: {e}")
        print("👉 Hãy chắc chắn bạn đã chạy 'ollama serve' và 'ollama pull qwen2.5:3b'")
        return

    # 2. Load dữ liệu
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file input '{INPUT_FILE}'.")
        return

    all_propositions = []

    # 3. Process từng chương
    # Nếu TEST_MODE = True, chỉ chạy chương đầu tiên
    chapters_to_process = data[6:8] if TEST_MODE else data

    for chapter in tqdm(chapters_to_process, desc="Total Progress"):
        props = process_chapter(chapter, chain_raw)
        all_propositions.extend(props)

    # 4. Save Output
    final_output_file = OUTPUT_FILE if not TEST_MODE else "test_behave_propositions.json"

    with open(final_output_file, "w", encoding="utf-8") as f:
        json.dump(all_propositions, f, indent=2, ensure_ascii=False)

    print(f"\n✅ HOÀN TẤT! Tổng cộng {len(all_propositions)} mệnh đề.")
    print(f"👉 Kết quả lưu tại: {final_output_file}")

    if TEST_MODE:
        print("\n--- PREVIEW KẾT QUẢ (5 dòng đầu) ---")
        print(json.dumps(all_propositions[:5], indent=2, ensure_ascii=False))
        print("\n💡 Nếu kết quả tốt, hãy chỉnh TEST_MODE = False để chạy full sách (sẽ tốn vài giờ).")


if __name__ == "__main__":
    main()
