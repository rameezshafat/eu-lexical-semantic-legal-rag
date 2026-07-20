"""
scripts/fetch_raw_pq.py — Fetch the literal original text of the 31
parliamentary-question-derived test queries (reviewer comment #32).

The gold standard's `query` field for these 31 queries is an author
paraphrase in plain conversational English; the `notes` field records the
official reference number(s) of the parliamentary question(s) it was
derived from (e.g. "E-002271/2024"), but not the literal text.

This fetches the primary (first-listed) reference's PDF from the European
Parliament's public document register (RegData), extracts the "Subject:"
line and the numbered question body via pypdf, and saves a mapping so a
later script can re-run retrieval against the raw wording. Network-
dependent; requires internet access to europarl.europa.eu.

URL pattern (reverse-engineered, confirmed working for both types):
  Written  (E-NNNNNN/YYYY): .../questions/ecrites/{Y}/{NNNNNN}/P10_QE({Y}){NNNNNN}_EN.pdf
  Priority (P-NNNNNN/YYYY): .../questions/ecrites/{Y}/{NNNNNN}/P10_QP({Y}){NNNNNN}_EN.pdf

Usage:
    python scripts/fetch_raw_pq.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import pypdf
from io import BytesIO

BASE = "https://www.europarl.europa.eu/RegData/questions/ecrites/{year}/{num}/P10_{code}({year}){num}_EN.pdf"


def _fetch_pdf_text(qtype: str, num: str, year: str) -> str | None:
    code = "QE" if qtype == "E" else "QP"
    url = BASE.format(year=year, num=num, code=code)
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            return None
        reader = pypdf.PdfReader(BytesIO(resp.content))
        return "\n".join(p.extract_text() for p in reader.pages)
    except Exception as e:
        print(f"    fetch failed for {qtype}-{num}/{year}: {e}")
        return None


def _extract_question_body(raw_text: str) -> str | None:
    """Strip PE-number header, submission metadata, subject line label,
    and footnotes; keep the "Subject: ..." line plus the numbered question
    body (or unnumbered body if the question has no sub-parts)."""
    m = re.search(r"Subject:\s*(.+?)(?=\nSubmitted:|\Z)", raw_text, re.S)
    if not m:
        return None
    body = m.group(1).strip()
    # Drop trailing footnote markers/lines (lines starting with a bare digit
    # followed by text, appended after the main body by the PDF extraction).
    body = re.sub(r"\n\d+\s+(Including|See|Cf\.).*", "", body, flags=re.S)
    return body.strip()


def main() -> None:
    gold = json.loads(Path("data/evaluation/gold_standard_test.json").read_text())
    queries = gold["queries"]

    results: dict[str, dict] = {}
    fetched, missing = 0, 0

    for q in queries:
        notes = q.get("notes", "")
        refs = re.findall(r"([EP])-(\d{6})/(\d{4})", notes)
        if not refs:
            continue
        qtype, num, year = refs[0]  # primary (first-listed) reference
        print(f"{q['query_id']}: fetching {qtype}-{num}/{year} …")
        raw = _fetch_pdf_text(qtype, num, year)
        if raw is None:
            print(f"    NOT FOUND")
            missing += 1
            continue
        body = _extract_question_body(raw)
        if body is None:
            print(f"    could not isolate question body from PDF text")
            missing += 1
            continue
        results[q["query_id"]] = {
            "reference": f"{qtype}-{num}/{year}",
            "all_references_in_notes": [f"{t}-{n}/{y}" for t, n, y in refs],
            "paraphrase": q["query"],
            "raw_pq_text": body,
        }
        fetched += 1
        time.sleep(0.5)  # be polite to the register

    print(f"\nFetched {fetched}, missing {missing}, out of "
          f"{fetched + missing} EP-derived queries.")

    out_path = Path("data/evaluation/raw_pq_text.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
