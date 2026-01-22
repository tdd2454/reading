import fitz  # PyMuPDF
import json

def build_hierarchical_toc(pdf_path):
    """
    Đọc Metadata TOC từ PDF và dựng thành cây phân cấp.
    """
    try:
        doc = fitz.open(pdf_path)
        toc = doc.get_toc() # [lvl, title, page, ...]
    except Exception as e:
        print(f"Lỗi mở file: {e}")
        return None

    if not toc:
        return None

    hierarchy = []
    # Stack dùng để theo dõi cha hiện tại ở từng cấp độ
    stack = {0: hierarchy} 

    for item in toc:
        level, title, page = item[0], item[1], item[2]
        
        # Tạo node hiện tại
        node = {
            "title": title,
            "start_page": page,
            "children": []
        }
        
        # Logic ghép cây
        parent_level = level - 1
        
        if parent_level in stack:
            parent_container = stack[parent_level]
            if isinstance(parent_container, list): # Root level
                parent_container.append(node)
            else: # Các level con
                parent_container["children"].append(node)
        
        stack[level] = node

    return hierarchy

def flatten_toc(toc_list):
    """
    Biến đổi cây phân cấp thành danh sách phẳng để dễ tính toán trang.
    """
    flat_list = []
    for item in toc_list:
        flat_list.append({
            "title": item["title"],
            "start_page": item["start_page"],
            "level": "Parent" if item.get("children") else "Leaf"
        })
        if item.get("children"):
            flat_list.extend(flatten_toc(item["children"]))
    return flat_list

def extract_content_by_ranges(pdf_path, toc_tree):
    """
    Dựa vào cây TOC, tính toán range trang và extract text.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # 1. Làm phẳng danh sách
    flat_toc = flatten_toc(toc_tree)
    # Sắp xếp lại theo số trang
    flat_toc.sort(key=lambda x: x["start_page"])
    
    structured_content = []
    
    # 2. Loop qua từng mục để lấy nội dung
    for i in range(len(flat_toc)):
        current_item = flat_toc[i]
        start_p = current_item["start_page"]
        
        # Xác định End Page
        if i < len(flat_toc) - 1:
            end_p = flat_toc[i+1]["start_page"] - 1
        else:
            end_p = total_pages
            
        if start_p > end_p: 
            end_p = start_p 

        # 3. Extract Text & Clean
        full_text = ""
        # PyMuPDF dùng 0-based index, metadata thường khớp với số trang PDF (1-based visual)
        # Ta trừ 1 để map về 0-based.
        for p_num in range(start_p - 1, end_p): 
            if p_num >= total_pages: break
            
            page = doc.load_page(p_num)
            text = page.get_text()
            
            # Cleaning Header/Footer (3 dòng đầu/cuối)
            lines = text.split('\n')
            if len(lines) > 10: 
                cleaned_lines = lines[3:-3] 
                text = '\n'.join(cleaned_lines)
            
            full_text += text + "\n"

        structured_content.append({
            "title": current_item["title"],
            "start_page": start_p,
            "end_page": end_p,
            "content": full_text.strip()
        })
        
        print(f"✅ Đã đọc: {current_item['title']} (Trang {start_p}-{end_p})")

    return structured_content

def main():
    # --- CẤU HÌNH ---
    pdf_file = r".\Input\sapolsky_behave.pdf" # Dùng r"" để tránh lỗi đường dẫn Windows
    
    print(f"--- BẮT ĐẦU XỬ LÝ FILE: {pdf_file} ---")

    # BƯỚC 1: Lấy Metadata TOC
    toc_tree = build_hierarchical_toc(pdf_file)

    if toc_tree:
        # In kiểm tra
        # print(json.dumps(toc_tree, indent=2, ensure_ascii=False))
        
        # Lưu file TOC JSON
        with open("toc_hierarchy.json", "w", encoding="utf-8") as f:
            json.dump(toc_tree, f, indent=2, ensure_ascii=False)
        print("✅ Đã lưu cấu trúc TOC vào file 'toc_hierarchy.json'")

        # BƯỚC 2: Extract nội dung dựa trên TOC vừa có
        print("\n--- ĐANG TRÍCH XUẤT NỘI DUNG CHI TIẾT ---")
        final_data = extract_content_by_ranges(pdf_file, toc_tree)

        # Lưu file Content JSON cuối cùng
        output_file = "behave_full_content.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        print(f"\n🎉 Hoàn tất! File '{output_file}' đã sẵn sàng cho bước Chunking.")
        
    else:
        print("❌ File không có Metadata TOC. Vui lòng dùng phương pháp khác.")

if __name__ == "__main__":
    main()