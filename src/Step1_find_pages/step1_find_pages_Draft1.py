import pdfplumber
import re
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
from pypdf import PdfReader, PdfWriter
from openai import OpenAI

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


def find_toc_range(pdf_path: Path, threshold: int = 35) -> Dict[str, Any]:
    scores_by_page: Dict[int, int] = {}

    with pdfplumber.open(str(pdf_path)) as pdf:
        for p, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            scores_by_page[p] = score_page_as_toc(text)

    # Candidate pages
    candidate_pages = sorted([p for p, s in scores_by_page.items() if s >= threshold])

    candidate_scored = [(p, scores_by_page[p]) for p in candidate_pages]

    candidate_scored_sorted = sorted(candidate_scored, key=lambda x: x[1], reverse=True)

    top5 = candidate_scored_sorted[:5]

    return {
        "pdf": str(pdf_path),
        "threshold": threshold,
        "top_5": top5
    }

# def make_one_page_pdf(src_pdf: Path, page_1based: int, out_pdf: Path) -> None:
#     reader = PdfReader(str(src_pdf))
#     if page_1based < 1 or page_1based > len(reader.pages):
#         raise ValueError(f"page {page_1based} out of range (1..{len(reader.pages)})")

#     writer = PdfWriter()
#     writer.add_page(reader.pages[page_1based - 1])

#     out_pdf.parent.mkdir(parents=True, exist_ok=True)
#     with out_pdf.open("wb") as f:
#         writer.write(f)

def make_mini_pdf(src_pdf: Path, pages_1based: List[int], out_pdf: Path) -> None:
    reader = PdfReader(str(src_pdf))
    n = len(reader.pages)

    writer = PdfWriter()
    for p in pages_1based:
        if p < 1 or p > n:
            raise ValueError(f"page {p} out of range (1..{n})")
        writer.add_page(reader.pages[p - 1])

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with out_pdf.open("wb") as f:
        writer.write(f)

def classify_page_as_toc(client: OpenAI, model: str, mini_pdf: Path, mapping: Dict[int, int]) -> Dict:
    # Upload mini PDF
    f = client.files.create(file=mini_pdf.open("rb"), purpose="user_data")

    prompt = {
        "task": "From the attached PDF (5 pages), pick exactly ONE page that is the true Table of Contents page.",
        "rules": [
            "Choose exactly ONE mini-PDF page number from the provided mapping.",
            "If none of the pages are Table of Contents, still choose the single best candidate and set confidence low.",
            "Return JSON only. No extra text."
        ],
        "page_mapping": mapping,
        "output_schema": {
            "mini_page_chosen": {"type": "integer"},
            "original_page_chosen": {"type": "integer"},
            "confidence": {"type": "number"},
            "rationale": {"type": "string"}
        }
    }

    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": f.id},
                {"type": "input_text", "text": json.dumps(prompt, ensure_ascii=False)},
            ],
        }],
    )

    text = resp.output_text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first line like ``` or ```json
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # drop last line ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except Exception:
        return {"error": "Model did not return valid JSON", "raw": resp.output_text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=str, help="Path to PDF")
    ap.add_argument("--threshold", type=int, default=35)
    # ap.add_argument("--min_block_len", type=int, default=2)
    ap.add_argument("--out", type=str, default="", help="Optional output JSON path")
    args = ap.parse_args()

    result = find_toc_range(Path(args.pdf), threshold=args.threshold)
    top5_pages = [p for p, s in result["top_5"]]

    print(top5_pages)

    # 1) create mini pdf with 5 pages
    mini_pdf = Path("tmp") / "top5_pages.pdf"
    make_mini_pdf(Path(args.pdf), top5_pages, mini_pdf)

    # 2) build mapping mini page -> original page (so result can be converted back)
    mapping: Dict[int, int] = {i + 1: top5_pages[i] for i in range(len(top5_pages))}

    print(mapping)
    # 3) call OpenAI once for the whole mini-pdf
    client = OpenAI(api_key="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")
    model = "gpt-4o-mini"

    decision = classify_page_as_toc(client, model, mini_pdf, mapping)
    print(decision)
    # for p in top5_pages:
    #     mini = Path("tmp_onepage") / f"page_{p}.pdf"
    #     make_one_page_pdf(Path(args.pdf), p, mini)
    #     r = classify_page_as_toc(client, model, mini, p)
    #     results.append(r)
    #     print(r)
    # # Print machine-readable output
    # print(json.dumps(result, ensure_ascii=False, indent=2))

    # # Optional save
    # if args.out:s
    #     Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
