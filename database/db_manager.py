import sqlite3
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_name="surveillance.db"):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        """Creates the alerts table if it doesn't exist."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                alert_type TEXT,
                person_id INTEGER,
                image_path TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_alert(self, alert_type, person_id, image_path):
        """Inserts a new alert record into the database."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO alerts (timestamp, alert_type, person_id, image_path)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, alert_type, person_id, image_path))
        conn.commit()
        conn.close()
        print(f"💾 Alert stored in Database: {alert_type}")