
from datetime import datetime
def parse_date_str(s: str) -> str:
    # Accept many formats; store ISO-8601 date
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%b %d, %Y", "%d %b %Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date().isoformat()
        except Exception:
            pass
    # Fallback: today
    return datetime.now().date().isoformat()
