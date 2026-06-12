"""
Shared utilities for the TCC-RAG pipeline.
"""
import re
import string
from dateutil.parser import parse


# ============================================================
# Text Normalization
# ============================================================

def normalize(text: str) -> str:
    """Normalize text: strip and collapse whitespace."""
    return " ".join(text.strip().split())


def normalize_for_eval(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace (for evaluation)."""
    s = str(s).lower()
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    return " ".join(s.split())


# ============================================================
# Time Extraction
# ============================================================

def extract_time(sentence: str) -> list:
    """
    Extract temporal expressions from a sentence.
    Supports: YYYY, YYYY-MM, YYYY-MM-DD, MM-DD-YYYY, English month names.
    Returns a list of normalized date strings.
    """
    pattern = (
        r'\b(?:\d{4}(?:-\d{1,2}(?:-\d{1,2})?)?|\d{1,2}-\d{1,2}-\d{4})\b'
        r'|'
        r'\b(?:\d{1,2}\s)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|'
        r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
        r'Nov(?:ember)?|Dec(?:ember)?)\s?\d{1,2}?(?:\s|,)?\s?\d{2,4}\b'
    )

    m = re.search(pattern, sentence)
    if not m:
        return []

    date_str = m.group()
    try:
        date_obj = parse(date_str)
    except Exception:
        return []

    # Determine granularity
    if re.fullmatch(r'\d{4}-\d{1,2}-\d{1,2}', date_str):
        return [f"{date_obj.year}-{date_obj.month:02d}-{date_obj.day:02d}"]
    elif re.fullmatch(r'\d{1,2}-\d{1,2}-\d{4}', date_str):
        return [f"{date_obj.year}-{date_obj.month:02d}-{date_obj.day:02d}"]
    elif re.fullmatch(r'\d{4}-\d{1,2}', date_str):
        return [f"{date_obj.year}-{date_obj.month:02d}"]
    elif re.fullmatch(r'\d{4}', date_str):
        return [f"{date_obj.year}"]
    else:
        # English month fallback
        parts = len(date_str.split(' '))
        if parts == 3:
            return [f"{date_obj.year}-{date_obj.month:02d}-{date_obj.day:02d}"]
        elif parts == 2:
            return [f"{date_obj.year}-{date_obj.month:02d}"]
        return [f"{date_obj.year}"]


def is_date_entity(text: str) -> bool:
    """
    Check if a string is a date/time expression.
    Returns True if it's a temporal entity that should be filtered out from NER results.
    """
    if not text or not isinstance(text, str):
        return False

    text_clean = text.strip().lower()

    # Common date patterns
    date_patterns = [
        r"^\d{4}$",                          # Year: 2011
        r"^\d{4}-\d{1,2}(-\d{1,2})?$",      # ISO: 2011-05-12
        r"^\d{1,2}\s+[a-z]{3,9}\s+\d{4}$",  # 25 April 2005
        r"^[a-z]{3,9}\s+\d{4}$",            # April 2005
        r"^\d{1,2}\s+[a-z]{3,9}$",          # 25 April
    ]
    for pat in date_patterns:
        if re.match(pat, text_clean):
            return True

    # Try parsing
    try:
        dt = parse(text_clean, fuzzy=False)
        if 1000 <= dt.year <= 2100:
            return True
    except Exception:
        pass

    # Month keywords (avoid false positives like "April Institute")
    months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    if any(m in text_clean for m in months):
        if not re.search(r"(institute|university|organization|council)", text_clean):
            return True

    return False
