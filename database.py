# database.py
import sqlite3          # Built-in SQLite database engine
import numpy as np      # For handling vectors (embeddings)
import json             # To store numpy arrays as JSON text
import os               # For file/directory operations

# Path where the SQLite database file will live
DB_PATH = "data/embeddings.db"


class FaceDatabase:
    def __init__(self, db_path=DB_PATH):
        # Save database path
        self.db_path = db_path  

        # Ensure the "data/" folder exists so SQLite file can be written
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Create tables if this is the first time running
        self._create_tables()


    def _connect(self):
        """Open a new SQLite database connection."""
        return sqlite3.connect(self.db_path)


    # -----------------------------------------------------
    # CREATE TABLES
    # -----------------------------------------------------
    def _create_tables(self):
        # Open database connection
        conn = self._connect()
        # Create cursor object to run SQL commands
        cur = conn.cursor() 

        # Create identity table: "persons"
        # - id autoincrements
        # - embedding stores a JSON string representing a 512-D vector
        cur.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding TEXT NOT NULL
            );
        """)

        # Create face observations table: "faces"
        # Stores every face embedding ever seen with timestamps
        cur.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                embedding TEXT NOT NULL,
                timestamp TEXT,
                FOREIGN KEY(person_id) REFERENCES persons(id)
            );
        """)

        # Save all database changes
        conn.commit()
        # Close connection
        conn.close()


    # -----------------------------------------------------
    # PERSON METHODS
    # -----------------------------------------------------
    def add_person(self, embedding: np.ndarray) -> int:
        """
        Create a new identity in the 'persons' table.
        Return the new person's ID (primary key).
        """

        conn = self._connect()
        cur = conn.cursor()

        # Convert numpy array → JSON string for saving
        emb_json = json.dumps(embedding.tolist())

        # Insert new person
        cur.execute("INSERT INTO persons (embedding) VALUES (?)", (emb_json,))
        conn.commit()

        # Retrieve database-assigned ID
        pid = cur.lastrowid

        conn.close()
        return pid


    def get_all_persons(self) -> dict:
        """
        Load ALL identities from the database.

        RETURNS:
            { person_id : embedding_vector }
        This format is optimized for fast lookup during face matching.
        """

        conn = self._connect()
        cur = conn.cursor()

        # Pull every stored identity and their embedding
        cur.execute("SELECT id, embedding FROM persons")
        rows = cur.fetchall()
        conn.close()

        persons = {}

        # Convert JSON → numpy vector for each entry
        for pid, emb_json in rows:
            vec = np.array(json.loads(emb_json), dtype=np.float32)
            persons[pid] = vec

        return persons


    def update_person_embedding(self, person_id: int, embedding: np.ndarray):
        """
        Updates the 'canonical' embedding for a person.
        Used for EMA (exponential moving average) refinement.
        """

        conn = self._connect()
        cur = conn.cursor()

        # Convert updated embedding → JSON string
        emb_json = json.dumps(embedding.tolist())

        # Update the row
        cur.execute(
            "UPDATE persons SET embedding = ? WHERE id = ?",
            (emb_json, person_id)
        )

        conn.commit()
        conn.close()


    # -----------------------------------------------------
    # FACE OBSERVATION LOGGING
    # -----------------------------------------------------
    def add_face_observation(self, person_id: int, embedding: np.ndarray, timestamp: str = ""):
        """
        Logs EVERY face seen — for debugging or future training.
        """

        conn = self._connect()
        cur = conn.cursor()

        # Convert embedding to JSON so SQLite can store it as text
        emb_json = json.dumps(embedding.tolist())

        # Insert observation
        cur.execute(
            "INSERT INTO faces (person_id, embedding, timestamp) VALUES (?, ?, ?)",
            (person_id, emb_json, timestamp)
        )

        conn.commit()
        conn.close()


    # -----------------------------------------------------
    # COUNT IDENTITIES
    # -----------------------------------------------------
    def count_persons(self) -> int:
        """Return how many unique identities exist in the database."""

        conn = self._connect()
        cur = conn.cursor()

        # COUNT(*) returns integer in first column
        cur.execute("SELECT COUNT(*) FROM persons")
        count = cur.fetchone()[0]

        conn.close()
        return count