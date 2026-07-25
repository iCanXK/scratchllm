#!/usr/bin/env python3
"""Download openly accessible PDF links from paper_links.json.

No publisher paywalls are bypassed. HTML-only resources are skipped.
"""
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlparse
import json, re, time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "papers"
OUT.mkdir(exist_ok=True)
items = json.loads((ROOT / "paper_links.json").read_text(encoding="utf-8"))

def safe_name(s):
    s = re.sub(r"[^A-Za-z0-9._ -]+", "", s).strip().replace(" ", "_")
    return s[:140] or "paper"

for item in items:
    url = item["url"]
    if not (url.lower().endswith(".pdf") or "/pdf/" in url.lower() or "arxiv.org/pdf/" in url.lower()):
        print(f"SKIP HTML: {item['number']:02d} {item['title']}")
        continue
    target = OUT / f"{item['number']:02d}_{safe_name(item['title'])}.pdf"
    if target.exists() and target.stat().st_size > 1000:
        print(f"EXISTS: {target.name}")
        continue
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (open-access research downloader)"})
        with urlopen(req, timeout=45) as r:
            data = r.read()
        if not data.startswith(b"%PDF"):
            print(f"NOT PDF: {item['title']}")
            continue
        target.write_bytes(data)
        print(f"DOWNLOADED: {target.name} ({len(data)//1024} KB)")
        time.sleep(0.4)
    except Exception as e:
        print(f"FAILED: {item['title']} - {e}")
