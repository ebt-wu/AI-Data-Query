import sqlite3

def clear_all_prompts():
    conn = sqlite3.connect("prompts.db")
    c = conn.cursor()
    c.execute("DELETE FROM prompts")
    conn.commit()
    conn.close()

def init_db():
    conn = sqlite3.connect("prompts.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

def save_prompt(content):
    conn = sqlite3.connect("prompts.db")
    c = conn.cursor()
    c.execute("INSERT INTO prompts (content) VALUES (?)", (content,))
    conn.commit()
    conn.close()

def get_all_prompts():
    conn = sqlite3.connect("prompts.db")
    c = conn.cursor()
    c.execute("SELECT id, content, created_at FROM prompts ORDER BY created_at DESC")
    results = c.fetchall()
    conn.close()
    return results

def get_prompt_by_id(prompt_id):
    conn = sqlite3.connect("prompts.db")
    c = conn.cursor()
    c.execute("SELECT content FROM prompts WHERE id = ?", (prompt_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None
