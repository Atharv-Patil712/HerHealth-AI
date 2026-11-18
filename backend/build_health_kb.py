import json
import sqlite3
from huggingface_hub import hf_hub_download
from app.database import DB_PATH

def build_knowledge_base():
    print("Downloading women-health-mini.jsonl from Hugging Face...")
    jsonl_path = hf_hub_download(
        repo_id="altaidevorg/women-health-mini",
        filename="women-health-mini.jsonl",
        repo_type="dataset",
    )

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS women_health_qa")
    cur.execute("""
        CREATE VIRTUAL TABLE women_health_qa USING fts5(
            question,
            answer,
            tokenize='porter unicode61'
        )
    """)

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            conv = item.get("conversations", [])
            if len(conv) >= 2:
                q = conv[0].get("content", "").strip()
                a = conv[1].get("content", "").strip()
                if q and a:
                    rows.append((q, a))

    cur.executemany("INSERT INTO women_health_qa (question, answer) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()
    print(f"Indexed {len(rows)} women's health Q&A pairs into {DB_PATH}!")


def ensure_knowledge_base():
    conn = sqlite3.connect(DB_PATH)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='women_health_qa'"
        ).fetchone()
    finally:
        conn.close()

    if not exists:
        build_knowledge_base()


if __name__ == "__main__":
    build_knowledge_base()