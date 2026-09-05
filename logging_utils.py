"""
Small helper for persisting per-question records (from basic_rag.py and
evaluate.py) as JSONL -- one JSON object per line, one file per kind of
run. JSONL keeps this append-only and streamable: you can tail a log
mid-run, and comparing two approaches later is just "load both files
into a dataframe."
"""
import json
import os
import time

import config


def _ensure_logs_dir():
    os.makedirs(config.LOGS_DIR, exist_ok=True)


def append_record(path: str, record: dict):
    """Appends one JSON object as a line to `path`, creating the file
    (and its parent directory) if needed. Adds a `logged_at` timestamp
    if the record doesn't already have one."""
    _ensure_logs_dir()
    record = dict(record)
    record.setdefault("logged_at", time.time())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_records(path: str):
    """Reads all JSONL records from `path`. Returns [] if the file
    doesn't exist yet (nothing logged there so far)."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
