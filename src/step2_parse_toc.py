import pdfplumber
import re
from pathlib import Path

PDF_PATH = Path("input/book.pdf")

PAGE_NUM_AT_END_RE = re.compile(r"^(?P<title>.*?)(?:\s?\.{2,}\s?|\s{2,})(?P<page>\d{1,4})\s*$")
# match: title .... 23  OR title<2+spaces>23

def parse_toc_from_pages(pages):
    toc = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for p in pages:
            page = pdf.pages[p-1]
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            for raw in text.splitlines():
                line = raw.strip()
                if not line:
                    continue
                m = PAGE_NUM_AT_END_RE.match(line)
                if not m:
                    continue
                title = m.group("title").strip(" .\t")
                page_num = int(m.group("page"))
                # lọc bớt rác: title quá ngắn hoặc quá dài
                if len(title) < 3 or len(title) > 140:
                    continue
                toc.append({"title": title, "page": page_num, "toc_page": p})
    return toc

def main():
    # Bạn chỉnh list này theo kết quả bước 1 (ví dụ TOC nằm trang 5-7)
    toc_pages = [5, 6, 7]

    toc = parse_toc_from_pages(toc_pages)

    print("Parsed TOC entries:", len(toc))
    for item in toc[:30]:
        print(f"{item['page']:>4}  {item['title']}   (from toc page {item['toc_page']})")

if __name__ == "__main__":
    main()
