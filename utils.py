import re
import unicodedata
from typing import Optional, Tuple


def normalize_txt(s: str) -> str:
    """
    Robust text normalization for Arabic/English text.
    - Unicode normalization (NFKC)
    - Replace NBSP with regular space
    - Convert Arabic digits to Western digits
    - Collapse multiple whitespaces
    - Strip leading/trailing whitespace
    """
    if not isinstance(s, str):
        s = str(s)
    
    # Unicode normalization
    s = unicodedata.normalize("NFKC", s)
    
    # Replace NBSP with regular space
    s = s.replace("\u00a0", " ")
    
    # Convert Arabic digits to Western digits
    arabic_to_western = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    s = s.translate(arabic_to_western)
    
    # Collapse multiple whitespaces and strip
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s


def safe_int_convert(x) -> int:
    """
    Convert quantity values to positive integers safely.
    Handles comma separators, spaces, and floating point values.
    """
    try:
        if isinstance(x, (int, float)):
            return abs(int(x))
        
        # Convert to string and clean
        val = str(x).replace(",", "").replace(" ", "")
        return abs(int(float(val)))
    except (ValueError, TypeError):
        return 0


def parse_month_generic(s: str, base: Tuple[str, str] = ("2024", "01")) -> Optional[int]:
    """
    Generic month parser that handles various formats and converts to integer offset.
    
    Supported formats:
    - "2025- Feb", "2025- Jan" (current format)
    - "Feb", "Jan", "Mar", etc. (bare month names, assumes current year)
    - "Aug-2025", "2024/07", "2025-08"
    - "2024-01", "2025-02" (ISO format)
    
    Args:
        s: Month string to parse
        base: Base year-month tuple for offset calculation (default: ("2024", "01"))
    
    Returns:
        Integer offset from base month, or None if parsing fails
    """
    if not isinstance(s, str):
        s = str(s)
    
    # Normalize the input
    s = normalize_txt(s).strip()
    if not s:
        return None
    
    # Month name mapping
    month_names = {
        "jan": "01", "january": "01",
        "feb": "02", "february": "02", 
        "mar": "03", "march": "03",
        "apr": "04", "april": "04",
        "may": "05",
        "jun": "06", "june": "06", "july": "07", "jul": "07",
        "aug": "08", "august": "08",
        "sep": "09", "sept": "09", "september": "09",
        "oct": "10", "october": "10",
        "nov": "11", "november": "11",
        "dec": "12", "december": "12"
    }
    
    year, month = None, None
    
    # Pattern 1: "2025- Feb", "2024- Jan"
    match = re.match(r'^(\d{4})-?\s*([a-zA-Z]+)$', s)
    if match:
        year = match.group(1)
        month_name = match.group(2).lower()
        month = month_names.get(month_name)
    
    # Pattern 2: "Feb-2025", "Aug-2024"
    if not month:
        match = re.match(r'^([a-zA-Z]+)-(\d{4})$', s)
        if match:
            month_name = match.group(1).lower()
            year = match.group(2)
            month = month_names.get(month_name)
    
    # Pattern 3: "2024/07", "2025/02"
    if not month:
        match = re.match(r'^(\d{4})[/\-](\d{1,2})$', s)
        if match:
            year = match.group(1)
            month = match.group(2).zfill(2)
    
    # Pattern 4: "2024-01", "2025-02" (ISO format)
    if not month:
        match = re.match(r'^(\d{4})-(\d{1,2})$', s)
        if match:
            year = match.group(1)
            month = match.group(2).zfill(2)
    
    # Pattern 5: Bare month names "Feb", "Jan" (assume current base year + 1 for future months)
    if not month:
        month_name = s.lower()
        if month_name in month_names:
            month = month_names[month_name]
            # For bare month names, use base year for past months, base+1 for future
            base_year = int(base[0])
            base_month = int(base[1])
            parsed_month = int(month)
            
            if parsed_month >= base_month:
                year = str(base_year)
            else:
                year = str(base_year + 1)
    
    # Validate parsed values
    if not year or not month:
        return None
    
    try:
        year_int = int(year)
        month_int = int(month)
        
        if not (1 <= month_int <= 12):
            return None
        
        # Calculate offset from base
        base_year_int = int(base[0])
        base_month_int = int(base[1])
        
        offset = (year_int - base_year_int) * 12 + (month_int - base_month_int)
        return offset
        
    except (ValueError, TypeError):
        return None


# Legacy month mapping for fallback compatibility
LEGACY_MONTH_MAP = {
    "Jan": "2024-01", "Feb": "2024-02", "Mar": "2024-03", "Apr": "2024-04",
    "May": "2024-05", "June": "2024-06", "Jul": "2024-07", "July": "2024-07",
    "Aug": "2024-08", "Sep": "2024-09", "Oct": "2024-10", "Nov": "2024-11", "Dec": "2024-12",
    "2025- Jan": "2025-01", "2025- Feb": "2025-02", "2025- Mar": "2025-03",
    "2025- Apr": "2025-04", "2025- May": "2025-05", "2025- Jun": "2025-06", "2025- Jul": "2025-07",
}


def parse_month_with_fallback(s: str, base: Tuple[str, str] = ("2024", "01")) -> Optional[int]:
    """
    Parse month using generic parser first, then fallback to legacy mapping.
    """
    # Try generic parser first
    result = parse_month_generic(s, base)
    if result is not None:
        return result
    
    # Fallback to legacy mapping
    normalized_s = normalize_txt(s)
    if normalized_s in LEGACY_MONTH_MAP:
        iso_date = LEGACY_MONTH_MAP[normalized_s]
        return parse_month_generic(iso_date, base)
    
    return None 