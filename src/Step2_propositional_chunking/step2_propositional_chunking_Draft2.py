import json
import time
import re
from pathlib import Path
from typing import List, Tuple, Optional, Any
from tqdm import tqdm
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CẤU HÌNH ---
INPUT_FILE = "behave_full_content.json"
OUTPUT_FILE = "behave_propositions.json"
MODEL_NAME = "qwen2.5:3b"  # Hoặc "qwen2.5:7b"
TEST_MODE = True  # True: Chỉ chạy thử 3 chunk đầu tiên để test. False: Chạy cả sách.

def strip_code_fences(s: str) -> str:
    # loại ```json ... ``` hoặc ``` ... ```
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def parse_json_array_of_strings(raw: str) -> Tuple[bool, Optional[List[str]], str]:
    """
    Trả về: (ok, propositions, error_message)
    """
    try:
        cleaned = strip_code_fences(raw)
        data = json.loads(cleaned)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return True, data, ""
        return False, None, f"JSON ok nhưng không phải list[str]. type={type(data)}"
    except Exception as e:
        return False, None, repr(e)

def dump_debug(debug_dir: str, chapter_title: str, window_idx: int,
               window_text: str, llm_raw: str, parse_err: str):
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
        encoding="utf-8"
    )
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
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])
    
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

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "TEXT:\n{text}\n\nReturn ONLY a JSON array of strings.")
        ])

        return prompt | llm  # <-- raw AIMessage

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

def process_chapter(chapter: dict, chain_raw, debug_dir: str = "debug_windows"):
    title = chapter.get("title", "Unknown")
    content = chapter.get("content", "") or ""
    windows = split_text_into_windows(content)

    propositions = []

    for i, window in enumerate(windows):
        try:
            msg = chain_raw.invoke({"text": window})
            llm_raw = getattr(msg, "content", str(msg))

            ok, props, err = parse_json_array_of_strings(llm_raw)
            if ok and props is not None:
                for p in props:
                    propositions.append({
                        "text": p,
                        "source_chapter": title,
                        "original_window_index": i
                    })
            else:
                dump_debug(debug_dir, title, i, window, llm_raw, err)
                propositions.append({
                    "text": window,
                    "source_chapter": title,
                    "note": f"raw_fallback: {err}"
                })

        except Exception as e:
            dump_debug(debug_dir, title, i, window, "", repr(e))
            propositions.append({
                "text": window,
                "source_chapter": title,
                "note": f"raw_fallback_exception: {repr(e)}"
            })

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
    chapters_to_process = data[6:7] if TEST_MODE else data
    
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