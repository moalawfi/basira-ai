import sqlite3

DB_PATH = "content_analyzer.db"

def get_connection(db_path=DB_PATH):
    return sqlite3.connect(db_path)

def init_db(db_path=DB_PATH):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            input_text TEXT,
            output_json TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            status TEXT NOT NULL,
            error_msg TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            metric_value TEXT NOT NULL,
            computed_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def drop_all_tables(db_path=DB_PATH):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute('DROP TABLE IF EXISTS results')
    cursor.execute('DROP TABLE IF EXISTS logs')
    cursor.execute('DROP TABLE IF EXISTS analytics_cache')
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    drop_all_tables()
    init_db()
    print("Database and tables initialized successfully.")