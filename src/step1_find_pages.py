import pdfplumber
import re
from pathlib import Path

PDF_PATH = Path("Input/sapolsky_behave.pdf")

TOC_KEYWORDS = [
    "table of contents",
    "contents",
    "mục lục",
]

PAGE_NUM_AT_END_RE = re.compile(r".*\s(\d{1,4})\s*$")  # title .... 23

def score_page_as_toc(text: str) -> int:
    if not text:
        return 0
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    low = text.lower()

    score = 0
    # keyword bonus
    for kw in TOC_KEYWORDS:
        if kw in low:
            score += 50

    # many lines end with a page number
    endnum = sum(1 for ln in lines if PAGE_NUM_AT_END_RE.match(ln))
    score += min(endnum, 40)  # cap

    # TOC pages usually have many short lines
    short_lines = sum(1 for ln in lines if 5 <= len(ln) <= 90)
    score += min(short_lines // 5, 20)

    return score

def main():
    candidates = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for p, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            score = score_page_as_toc(text)
            if score >= 35:  # threshold, you can tune
                candidates.append((p, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    print("TOC candidates (page, score):")
    for p, s in candidates[:15]:
        print(p, s)

if __name__ == "__main__":
    main()
