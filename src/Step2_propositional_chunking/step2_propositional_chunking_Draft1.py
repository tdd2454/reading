import json
import time
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

def process_chapter(chapter_data, chain):
    """
    Xử lý một chương: Cắt nhỏ -> Gửi LLM -> Gom lại.
    """
    title = chapter_data['title']
    raw_content = chapter_data['content']
    
    # 1. Cắt nhỏ content thành các windows
    windows = split_text_into_windows(raw_content)
    
    # if TEST_MODE:
    #     print(f"   [TEST MODE] Chỉ xử lý 3 đoạn đầu của chương '{title}'...")
    #     windows = windows[:3]

    chapter_propositions = []
    
    # 2. Loop qua từng window để xử lý
    for i, window in enumerate(tqdm(windows, desc=f"   Processing {title}", leave=False)):
        try:
            # Gọi LLM
            result = chain.invoke({"text": window})
            
            # Result kỳ vọng là list strings: ["fact 1", "fact 2"]
            if isinstance(result, list):
                # Lưu thêm metadata nguồn gốc
                for prop in result:
                    chapter_propositions.append({
                        "text": prop,
                        "source_chapter": title,
                        "original_window_index": i
                    })
            else:
                # Fallback nếu LLM trả về format lạ (hiếm gặp với Qwen)
                chapter_propositions.append({"text": window, "source_chapter": title, "note": "raw_fallback"})
                
        except Exception as e:
            print(f"   ⚠️ Lỗi chunk {i}: {e}")
            continue
            
    return chapter_propositions

def main():
    print(f"--- BẮT ĐẦU PROPOSITIONAL CHUNKING (Model: {MODEL_NAME}) ---")
    
    # 1. Setup
    try:
        chain = setup_llm_chain()
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
        props = process_chapter(chapter, chain)
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