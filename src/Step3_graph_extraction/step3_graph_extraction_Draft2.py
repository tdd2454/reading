import json
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# =============================================================================
# STEP 3 - GRAPH EXTRACTION (OPTIMIZED 3-LAYER MICRO-CONTEXT)
#  Layer A: Adaptive window (decide when to pull prev/next)
#  Layer B: Topic coherence (select best neighbor sentences by keyword overlap)
#  Layer C: Length cap (keep CURRENT, clip side context)
# =============================================================================

# --- CONFIG ---
INPUT_FILE = "test_behave_propositions.json"
OUTPUT_FILE = "behave_graph_triplets.json"
MODEL_NAME = "qwen2.5:3b"

DEBUG_LIMIT = None         # e.g. 10 for quick test; None for all
MIN_TEXT_LEN = 10

# Micro-context tuning
MAX_WINDOW = 3             # pool: i-MAX_WINDOW .. i+MAX_WINDOW
MAX_SIDE_ITEMS = 2         # add up to N extra coherent items per side
MAX_CONTEXT_CHARS = 1000   # total context chars cap
SIDE_CLIP_CHARS = 280      # clip each side item

# Filters
PLACEHOLDER_TAILS = {"an event", "event", "something", "someone", "it", "this", "that"}
META_DROP_PATTERNS = (
    "as will be discussed", "will be discussed in chapter", "as will be shown",
    "as will be described", "as we will see"
)

# --- PROMPT ---
system_prompt = """
You are an expert Knowledge Graph Extractor.
Goal: Extract structured triplets (Head -> Relation -> Tail) from the provided text.

RULES:
1. ENTITY TYPES: "Agent", "Object", "Concept", "Event".
2. RELATIONS: Use concise, active verbs. Prefer stable, KG-friendly relations.
3. OUTPUT FORMAT: Return a JSON object with a single key "triplets" strictly like this:
   {{
     "triplets": [
       {{ "head": "A", "head_type": "Agent", "relation": "CAUSES", "tail": "B", "tail_type": "Event" }}
     ]
   }}
4. If no clear factual relation exists, return:
   {{ "triplets": [] }}
5. IMPORTANT:
   - Use phrases from CURRENT as head/tail when possible.
   - Do NOT invent facts not supported by the text.
6. NO EXPLANATION. OUTPUT ONLY JSON.
"""

# =============================================================================
# 3-LAYER MICRO-CONTEXT BUILDER
# =============================================================================

# minimal English stopwords for overlap scoring
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "while",
    "of", "to", "in", "on", "for", "with", "as", "at", "by", "from", "into",
    "over", "under", "between", "about", "than", "so", "such",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "this", "that", "these", "those", "it", "its",
    "he", "she", "they", "them", "his", "her", "their",
}

PRONOUN_STARTS = (
    "he ", "she ", "it ", "this ", "that ", "these ", "those ", "such ",
    "one ", "we ", "they ", "his ", "her ", "their "
)

@dataclass
class ContextItem:
    tag: str
    text: str
    score: float

def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _tokenize_keywords(s: str) -> List[str]:
    s = (s or "").lower()
    words = re.findall(r"[a-z0-9']+", s)
    return [w for w in words if len(w) >= 3 and w not in _STOPWORDS]

def _overlap_score(a: str, b: str) -> float:
    A = set(_tokenize_keywords(a))
    B = set(_tokenize_keywords(b))
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    jacc = inter / union

    # small bigram bonus for cohesion
    def bigrams(tokens: List[str]) -> set:
        return set(zip(tokens, tokens[1:])) if len(tokens) >= 2 else set()

    Ab = bigrams(_tokenize_keywords(a))
    Bb = bigrams(_tokenize_keywords(b))
    bigram_bonus = 0.0
    if Ab and Bb:
        bigram_bonus = 0.15 * (len(Ab & Bb) / max(1, len(Ab | Bb)))

    return jacc + bigram_bonus

def needs_prev(current: str) -> bool:
    t = _normalize_space(current).lower()
    if not t:
        return False
    if any(p in t for p in META_DROP_PATTERNS):
        return True
    if t.startswith(PRONOUN_STARTS):
        return True
    if "such scientists" in t or "the last term" in t or "this term" in t:
        return True
    return False

def needs_next(current: str) -> bool:
    t = _normalize_space(current).lower()
    if not t:
        return False
    if t.startswith(("this ", "that ", "these ", "such ")):
        return True
    if t.endswith(("...", ":", "—")):
        return True
    return False

def is_hard_drop(current: str) -> bool:
    """
    Drop fragments that tend to generate garbage nodes like "will be discussed".
    You can loosen/tighten later.
    """
    t = _normalize_space(current).lower()
    if any(p in t for p in META_DROP_PATTERNS):
        return True
    if len(t) < 12:
        return True
    return False

def _clip(text: str, max_chars: int) -> str:
    text = _normalize_space(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "..."

def build_micro_context(
    propositions: list,
    i: int,
    max_window: int = MAX_WINDOW,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    max_side_items: int = MAX_SIDE_ITEMS,
    side_clip_chars: int = SIDE_CLIP_CHARS,
) -> Tuple[str, str]:
    """
    Returns:
      - context_for_llm: labeled multi-line context with PREV/NEXT/CURRENT
      - current_text_clean: the CURRENT sentence (for provenance)
    """
    cur = _normalize_space(propositions[i].get("text", ""))
    if not cur:
        return "", ""

    want_prev = needs_prev(cur)
    want_next = needs_next(cur)

    n = len(propositions)
    lo = max(0, i - max_window)
    hi = min(n - 1, i + max_window)

    pool: List[Tuple[int, str]] = []
    for j in range(lo, hi + 1):
        if j == i:
            continue
        txt = _normalize_space(propositions[j].get("text", ""))
        if txt:
            pool.append((j, txt))

    scored: List[ContextItem] = []
    for j, txt in pool:
        sc = _overlap_score(cur, txt)
        tag = "PREV_COH" if j < i else "NEXT_COH"
        scored.append(ContextItem(tag=tag, text=txt, score=sc))

    prev_candidates = sorted([c for c in scored if c.tag == "PREV_COH"], key=lambda x: x.score, reverse=True)
    next_candidates = sorted([c for c in scored if c.tag == "NEXT_COH"], key=lambda x: x.score, reverse=True)

    selected: List[ContextItem] = []

    # Prefer immediate neighbors when needed
    if want_prev and i - 1 >= 0:
        p1 = _normalize_space(propositions[i - 1].get("text", ""))
        if p1:
            selected.append(ContextItem(tag="PREV", text=p1, score=_overlap_score(cur, p1)))

    if want_next and i + 1 < n:
        n1 = _normalize_space(propositions[i + 1].get("text", ""))
        if n1:
            selected.append(ContextItem(tag="NEXT", text=n1, score=_overlap_score(cur, n1)))

    # Add best coherent items per side (no duplicates)
    def _add_best(cands: List[ContextItem], tag: str, k: int):
        added = 0
        for c in cands:
            if added >= k:
                break
            if c.score <= 0:
                continue
            if any(c.text == s.text for s in selected):
                continue
            selected.append(ContextItem(tag=tag, text=c.text, score=c.score))
            added += 1

    if want_prev:
        _add_best(prev_candidates, "PREV_COH", max_side_items)
    if want_next:
        _add_best(next_candidates, "NEXT_COH", max_side_items)

    # Build labeled context; clip side items
    prev_parts = [s for s in selected if s.tag.startswith("PREV")]
    next_parts = [s for s in selected if s.tag.startswith("NEXT")]

    prev_parts = sorted(prev_parts, key=lambda x: x.score, reverse=True)
    next_parts = sorted(next_parts, key=lambda x: x.score, reverse=True)

    prev_lines = [f"{s.tag}: {_clip(s.text, side_clip_chars)}" for s in prev_parts]
    next_lines = [f"{s.tag}: {_clip(s.text, side_clip_chars)}" for s in next_parts]

    current_line = f"CURRENT: {cur}"
    context = "\n".join(prev_lines + [current_line] + next_lines).strip()

    # Length cap: keep CURRENT always, add best sides while fits
    if len(context) > max_context_chars:
        kept = [current_line]
        ranked = prev_parts + next_parts
        ranked = sorted(ranked, key=lambda x: x.score, reverse=True)

        for s in ranked:
            line = f"{s.tag}: {_clip(s.text, side_clip_chars)}"
            tentative = "\n".join([line] + kept) if s.tag.startswith("PREV") else "\n".join(kept + [line])
            if len(tentative) <= max_context_chars:
                kept = [line] + kept if s.tag.startswith("PREV") else kept + [line]

        context = "\n".join(kept).strip()

    return context, cur

# =============================================================================
# LLM CHAINS + JSON PARSING
# =============================================================================

def build_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Text:\n{text}")
    ])

def setup_chains():
    """
    Create two chains:
      - chain_json: tries format='json' if supported
      - chain_plain: plain output fallback
    """
    prompt = build_prompt()

    chain_json = None
    try:
        llm_json = ChatOllama(model=MODEL_NAME, temperature=0, format="json")
        chain_json = prompt | llm_json
    except TypeError:
        chain_json = None

    llm_plain = ChatOllama(model=MODEL_NAME, temperature=0)
    chain_plain = prompt | llm_plain
    return chain_json, chain_plain

def _looks_like_empty_object_output(content) -> bool:
    if content is None:
        return True
    if isinstance(content, dict):
        return len(content) == 0 or ("triplets" not in content)
    if isinstance(content, str):
        s = content.strip()
        if s in ("{}", ""):
            return True
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and ("triplets" not in obj):
                return True
        except Exception:
            pass
    return False

def extract_triplets_safe(llm_output):
    """
    Returns (triplets_list, status) where status in {"ok","empty","parse_fail"}.
    """
    # dict
    if isinstance(llm_output, dict):
        if "triplets" in llm_output and isinstance(llm_output["triplets"], list):
            triplets = llm_output["triplets"]
            return (triplets, "ok" if triplets else "empty")
        for _, v in llm_output.items():
            if isinstance(v, list):
                return (v, "ok" if v else "empty")
        return ([], "empty")

    # list
    if isinstance(llm_output, list):
        return (llm_output, "ok" if llm_output else "empty")

    # string
    if isinstance(llm_output, str):
        text = llm_output.strip()
        if not text:
            return ([], "empty")

        text = text.replace("```json", "").replace("```", "").strip()

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                if "triplets" in obj and isinstance(obj["triplets"], list):
                    triplets = obj["triplets"]
                    return (triplets, "ok" if triplets else "empty")
                return ([], "empty")
            if isinstance(obj, list):
                return (obj, "ok" if obj else "empty")
        except Exception:
            pass

        # fallback: find array
        match = re.search(r"\[[\s\S]*?\]", text)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return (data, "ok" if data else "empty")
            except Exception:
                return ([], "parse_fail")

        return ([], "parse_fail")

    return ([], "parse_fail")

# =============================================================================
# NORMALIZATION + DEDUPE
# =============================================================================

def normalize_relation(rel: str) -> str:
    rel = (rel or "").strip().upper()
    rel = re.sub(r"\s+", "_", rel)
    rel = re.sub(r"[^A-Z0-9_]", "", rel)
    return rel

def normalize_triplet(t):
    """
    Minimal validation + normalization.
    Returns normalized dict or None.
    """
    if not isinstance(t, dict):
        return None

    head = str(t.get("head", "")).strip()
    rel  = normalize_relation(str(t.get("relation", "")))
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

def make_dedupe_key(t, chapter: str):
    return (t["head"].lower(), t["relation"].lower(), t["tail"].lower(), str(chapter).lower())

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("--- BẮT ĐẦU EXTRACT GRAPH (OPTIMIZED 3-LAYER MICRO-CONTEXT) ---")

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
    dropped_weak = 0

    total = len(propositions)
    iterator = tqdm(range(total))

    for idx in iterator:
        if DEBUG_LIMIT is not None and idx >= DEBUG_LIMIT:
            break

        prop = propositions[idx]
        note = str(prop.get("note", "") or "")
        chapter = prop.get("source_chapter", "Unknown")

        # skip step2 "raw_fallback" if you want
        if "raw_fallback" in note:
            continue

        context_for_llm, current_text = build_micro_context(
            propositions,
            idx,
            max_window=MAX_WINDOW,
            max_context_chars=MAX_CONTEXT_CHARS,
            max_side_items=MAX_SIDE_ITEMS,
            side_clip_chars=SIDE_CLIP_CHARS,
        )

        if not current_text:
            continue

        if len(current_text) < MIN_TEXT_LEN:
            continue

        if is_hard_drop(current_text):
            dropped_weak += 1
            continue

        # Invoke: try JSON chain first (if exists), fallback to plain if output looks empty/bad
        try:
            raw_content = None
            if chain_json is not None:
                raw_result = chain_json.invoke({"text": context_for_llm})
                raw_content = raw_result.content

            if raw_content is None or _looks_like_empty_object_output(raw_content):
                raw_result = chain_plain.invoke({"text": context_for_llm})
                raw_content = raw_result.content
                fallback_retry += 1

            triplets, status = extract_triplets_safe(raw_content)

            if status == "parse_fail":
                parse_fail += 1
                # keep log compact
                print(f"\n⚠️ Item {idx}: Parse FAIL.")
                print(f"   CURRENT: {current_text[:160]}...")
                print(f"   RAW    : {str(raw_content)[:220]}...")
                continue

            if status == "empty":
                empty_count += 1
                continue

            # status == ok
            for t in triplets:
                norm = normalize_triplet(t)
                if not norm:
                    bad_schema += 1
                    continue

                key = make_dedupe_key(norm, chapter)
                if key in seen:
                    continue
                seen.add(key)

                # provenance: keep CURRENT only (clean)
                norm["source_text"] = current_text
                norm["source_chapter"] = chapter
                all_triplets.append(norm)

        except Exception as e:
            print(f"\n❌ Lỗi hệ thống tại item {idx}: {e}")
            continue

    if all_triplets:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_triplets, f, indent=2, ensure_ascii=False)

        print(f"\n✅ THÀNH CÔNG! Extract được {len(all_triplets)} triplets (deduped).")
        print(
            "Stats: "
            f"parse_fail={parse_fail}, empty={empty_count}, bad_schema={bad_schema}, "
            f"fallback_retry={fallback_retry}, dropped_weak={dropped_weak}"
        )
        print(json.dumps(all_triplets[:8], indent=2, ensure_ascii=False))
    else:
        print("\n❌ Chưa thu được triplet nào.")
        print(
            "Stats: "
            f"parse_fail={parse_fail}, empty={empty_count}, bad_schema={bad_schema}, "
            f"fallback_retry={fallback_retry}, dropped_weak={dropped_weak}"
        )
        print("Gợi ý: nếu empty cao => câu quá 'định nghĩa/tu từ' hoặc prompt đang quá strict;")
        print("nếu parse_fail cao => model không bám JSON contract (thử model khác hoặc siết prompt).")

if __name__ == "__main__":
    main()
