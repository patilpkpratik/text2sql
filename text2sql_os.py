import sqlite3
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from difflib import SequenceMatcher

# Path to the database
DB_PATH = "/Users/abhijitsen/Downloads/text-to-sql/Chinook.db"
# Open-source text-to-SQL model (example: t5-small-text-to-sql)
MODEL_NAME = "tscholak/optical-small-100k"

def get_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    schema = {}
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        col_defs = ", ".join([f"{col[1]} {col[2]}" for col in columns])
        schema[table_name] = f"CREATE TABLE {table_name} ({col_defs});"
    conn.close()
    return schema


# RAG: Retrieve relevant schema context for the question
def retrieve_relevant_schema(question, schema_dict, top_k=3):
    scores = []
    for table, ddl in schema_dict.items():
        score = SequenceMatcher(None, question.lower(), ddl.lower()).ratio()
        scores.append((score, ddl))
    scores.sort(reverse=True)
    relevant_ddls = [ddl for _, ddl in scores[:top_k]]
    return " ".join(relevant_ddls)


def text_to_sql(question, schema_dict):
    relevant_schema = retrieve_relevant_schema(question, schema_dict)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    prompt = f"Given an input question, convert it to a SQL query. No pre-amble. Please do not return anything else apart from the SQL query, no prefix aur suffix quotes, no sql keyword, nothing please: {question} \n {relevant_schema}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256)
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql.strip()


if __name__ == "__main__":
    import sys
    question = input().strip() if len(sys.argv) == 1 else " ".join(sys.argv[1:])
    schema_dict = get_schema(DB_PATH)
    sql_query = text_to_sql(question, schema_dict)
    print(sql_query)
