import sqlite3

def initialize_database():
    conn = sqlite3.connect('oral_cancer.db')  # Replace with your database file
    cursor = conn.cursor()

    # Create the patient_records table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patient_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            state TEXT,
            district TEXT,
            doctor TEXT,
            timestamp TEXT,
            symptoms TEXT,
            prediction TEXT,
            confidence REAL,
            image_path TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def check_patient_records():
    conn = sqlite3.connect('oral_cancer.db')  # Replace with your database file
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM patient_records')
    records = cursor.fetchall()

    for record in records:
        print(record)

    conn.close()

if __name__ == "__main__":
    initialize_database()
    check_patient_records()