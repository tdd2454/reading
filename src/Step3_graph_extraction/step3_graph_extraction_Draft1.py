import json
import re
from tqdm import tqdm
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- CẤU HÌNH ---
INPUT_FILE = "test_behave_propositions.json"
OUTPUT_FILE = "behave_graph_triplets.json"
MODEL_NAME = "qwen2.5:3b"

DEBUG_LIMIT = None      # ví dụ: 10 để test nhanh; None để chạy hết
MIN_TEXT_LEN = 10

PLACEHOLDER_TAILS = {"an event", "event", "something", "someone", "it", "this", "that"}

# --- PROMPT ---
system_prompt = """
You are an expert Knowledge Graph Extractor.
Goal: Extract structured triplets (Head -> Relation -> Tail) from the provided text.

RULES:
1. ENTITY TYPES: "Agent", "Object", "Concept", "Event".
2. RELATIONS: Use concise, active verbs.
3. OUTPUT FORMAT: Return a JSON object with a single key "triplets" strictly like this:
   {{
     "triplets": [
       {{ "head": "A", "head_type": "Agent", "relation": "CAUSES", "tail": "B", "tail_type": "Event" }}
     ]
   }}
4. If no clear factual relation exists, return:
   {{ "triplets": [] }}
5. NO EXPLANATION. OUTPUT ONLY JSON.
"""

def build_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Text: {text}")
    ])

def setup_chains():
    """
    Tạo 2 chain:
    - chain_json: cố gắng ép JSON mode (nếu supported)
    - chain_plain: không ép JSON mode (fallback)
    """
    prompt = build_prompt()

    # Chain JSON (nếu driver hỗ trợ)
    try:
        llm_json = ChatOllama(model=MODEL_NAME, temperature=0, format="json")
        chain_json = prompt | llm_json
    except TypeError:
        chain_json = None

    # Chain không ép JSON (fallback chắc chắn chạy)
    llm_plain = ChatOllama(model=MODEL_NAME, temperature=0)
    chain_plain = prompt | llm_plain

    return chain_json, chain_plain

def _looks_like_empty_object_output(content) -> bool:
    """
    Nhiều khi Ollama/langchain trả {} (dict rỗng) hoặc '{}' (string)
    trong JSON mode. Ta coi đây là output "trống/không hữu ích" để retry.
    """
    if content is None:
        return True

    if isinstance(content, dict):
        return len(content) == 0 or ("triplets" not in content)

    if isinstance(content, str):
        s = content.strip()
        if s in ("{}", ""):
            return True
        # đôi khi là JSON object nhưng không có triplets
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and ("triplets" not in obj):
                return True
        except Exception:
            pass

    return False

def extract_triplets_safe(llm_output):
    """
    Trả về tuple: (triplets_list, status)
    status ∈ {"ok", "empty", "parse_fail"}
    """
    # Case 1: dict
    if isinstance(llm_output, dict):
        if "triplets" in llm_output and isinstance(llm_output["triplets"], list):
            triplets = llm_output["triplets"]
            return (triplets, "ok" if triplets else "empty")

        # fallback: tìm value nào là list
        for _, value in llm_output.items():
            if isinstance(value, list):
                return (value, "ok" if value else "empty")

        return ([], "empty")

    # Case 2: list trực tiếp
    if isinstance(llm_output, list):
        return (llm_output, "ok" if llm_output else "empty")

    # Case 3: string
    if isinstance(llm_output, str):
        text = llm_output.strip()
        if not text:
            return ([], "empty")

        text = text.replace("```json", "").replace("```", "").strip()

        # thử parse toàn bộ string như JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                if "triplets" in obj and isinstance(obj["triplets"], list):
                    triplets = obj["triplets"]
                    return (triplets, "ok" if triplets else "empty")
                # object khác -> coi như empty
                return ([], "empty")
            if isinstance(obj, list):
                return (obj, "ok" if obj else "empty")
        except Exception:
            pass

        # fallback: bắt array [...]
        match = re.search(r"\[[\s\S]*?\]", text)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return (data, "ok" if data else "empty")
            except Exception:
                return ([], "parse_fail")

        return ([], "parse_fail")

    # Unknown type
    return ([], "parse_fail")

def normalize_triplet(t):
    """
    Validate + normalize tối thiểu để dùng cho KG.
    Trả dict chuẩn hoặc None nếu rác.
    """
    if not isinstance(t, dict):
        return None

    head = str(t.get("head", "")).strip()
    rel  = str(t.get("relation", "")).strip()
    tail = str(t.get("tail", "")).strip()

    if not head or not rel or not tail:
        return None

    if tail.lower() in PLACEHOLDER_TAILS:
        return None

    head_type = str(t.get("head_type", "")).strip() or None
    tail_type = str(t.get("tail_type", "")).strip() or None

    out = {"head": head, "relation": rel, "tail": tail}
    if head_type:
        out["head_type"] = head_type
    if tail_type:
        out["tail_type"] = tail_type
    return out

def make_dedupe_key(t, chapter):
    return (t["head"].lower(), t["relation"].lower(), t["tail"].lower(), str(chapter).lower())

def main():
    print("--- BẮT ĐẦU EXTRACT GRAPH (DEBUG + FORCE PARSE MODE) ---")

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            propositions = json.load(f)
    except FileNotFoundError:
        print(f"❌ Thiếu file {INPUT_FILE}")
        return

    chain_json, chain_plain = setup_chains()

    all_triplets = []
    seen = set()

    parse_fail = 0
    empty_count = 0
    bad_schema = 0
    fallback_retry = 0

    for i, prop in enumerate(tqdm(propositions)):
        if DEBUG_LIMIT is not None and i >= DEBUG_LIMIT:
            break

        text = prop.get("text", "") or ""
        note = str(prop.get("note", "") or "")

        if len(text) < MIN_TEXT_LEN:
            continue
        if "raw_fallback" in note:
            continue

        chapter = prop.get("source_chapter", "Unknown")

        try:
            # 1) Thử chain_json trước (nếu có)
            raw_content = None
            if chain_json is not None:
                raw_result = chain_json.invoke({"text": text})
                raw_content = raw_result.content

            # 2) Nếu output có mùi "{}" hoặc không có triplets => retry bằng chain_plain
            if raw_content is None or _looks_like_empty_object_output(raw_content):
                raw_result = chain_plain.invoke({"text": text})
                raw_content = raw_result.content
                fallback_retry += 1

            triplets, status = extract_triplets_safe(raw_content)

            if status == "parse_fail":
                parse_fail += 1
                print(f"\n⚠️ Item {i}: Parse FAIL (không đọc được JSON/list).")
                print(f"   TEXT: {text[:120]}...")
                print(f"   RAW : {str(raw_content)[:200]}...")
                continue

            if status == "empty":
                # Đây KHÔNG phải lỗi. Chỉ là model trả không có triplets.
                empty_count += 1
                # Nếu bạn muốn log ít, comment block này lại:
                # print(f"\nℹ️ Item {i}: JSON hợp lệ nhưng triplets rỗng.")
                # print(f"   RAW : {str(raw_content)[:120]}...")
                continue

            # status == "ok"
            for t in triplets:
                norm = normalize_triplet(t)
                if not norm:
                    bad_schema += 1
                    continue

                key = make_dedupe_key(norm, chapter)
                if key in seen:
                    continue
                seen.add(key)

                norm["source_text"] = text
                norm["source_chapter"] = chapter
                all_triplets.append(norm)

        except Exception as e:
            print(f"\n❌ Lỗi hệ thống tại item {i}: {e}")
            continue

    if all_triplets:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_triplets, f, indent=2, ensure_ascii=False)

        print(f"\n✅ THÀNH CÔNG! Extract được {len(all_triplets)} triplets (deduped).")
        print(f"Stats: parse_fail={parse_fail}, empty={empty_count}, bad_schema={bad_schema}, fallback_retry={fallback_retry}")
        print(json.dumps(all_triplets[:5], indent=2, ensure_ascii=False))
    else:
        print("\n❌ Chưa thu được triplet nào.")
        print(f"Stats: parse_fail={parse_fail}, empty={empty_count}, bad_schema={bad_schema}, fallback_retry={fallback_retry}")
        print("Gợi ý: nếu empty quá cao, model/prompt đang quá 'khắt khe' hoặc model yếu; nếu parse_fail cao, output đang không theo JSON contract.")

if __name__ == "__main__":
    main()
