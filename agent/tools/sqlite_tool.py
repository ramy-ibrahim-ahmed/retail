import sqlite3
import pandas as pd

DB_PATH = "data/northwind.sqlite"


def get_schema():
    """Returns the schema of key tables for the LLM context."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    schema_str = ""
    tables = ["Categories", "customers", "orders", "order_items", "products"]
    for table in tables:
        cursor.execute(f"PRAGMA table_info('{table}')")
        columns = [row[1] for row in cursor.fetchall()]
        schema_str += f"Table: {table}\nColumns: {', '.join(columns)}\n\n"

    conn.close()
    return schema_str


def run_query(query: str):
    """Executes SQL and returns a dataframe or error string."""
    try:
        conn = sqlite3.connect(DB_PATH)
        if ";" in query.strip()[:-1]:
            return None, "Error: Multiple statements not allowed."

        df = pd.read_sql_query(query, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)
