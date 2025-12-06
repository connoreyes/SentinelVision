# database.py
import sqlite3
import numpy as np
import json
import os

DB_PATH = "data/embeddings.db"


class FaceDatabase:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._create_tables()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    # -----------------------------------------------------
    # CREATE TABLES
    # -----------------------------------------------------
    def _create_tables(self):
        conn = self._connect()
        cur = conn.cursor()

        # Canonical identity table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding TEXT NOT NULL
            );
        """)

        # Observation history
        cur.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                embedding TEXT NOT NULL,
                timestamp TEXT,
                FOREIGN KEY(person_id) REFERENCES persons(id)
            );
        """)

        conn.commit()
        conn.close()

    # -----------------------------------------------------
    # PERSON METHODS
    # -----------------------------------------------------
    def add_person(self, embedding: np.ndarray) -> int:
        """Add a new identity and return its ID."""
        conn = self._connect()
        cur = conn.cursor()

        emb_json = json.dumps(embedding.tolist())
        cur.execute("INSERT INTO persons (embedding) VALUES (?)", (emb_json,))
        conn.commit()

        pid = cur.lastrowid
        conn.close()
        return pid

    def get_all_persons(self) -> dict:
        """
        RETURNS:
            { person_id : embedding_vector }
        MUCH FASTER than returning a list.
        """
        conn = self._connect()
        cur = conn.cursor()

        cur.execute("SELECT id, embedding FROM persons")
        rows = cur.fetchall()
        conn.close()

        persons = {}
        for pid, emb_json in rows:
            vec = np.array(json.loads(emb_json), dtype=np.float32)
            persons[pid] = vec

        return persons

    def update_person_embedding(self, person_id: int, embedding: np.ndarray):
        """Updates a stored identity's canonical embedding (EMA refinement)."""
        conn = self._connect()
        cur = conn.cursor()

        emb_json = json.dumps(embedding.tolist())
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
        """Stores each face seen — improves debugging and training later."""
        conn = self._connect()
        cur = conn.cursor()

        emb_json = json.dumps(embedding.tolist())
        cur.execute(
            "INSERT INTO faces (person_id, embedding, timestamp) VALUES (?, ?, ?)",
            (person_id, emb_json, timestamp)
        )

        conn.commit()
        conn.close()


    def count_persons(self) -> int:
        """Return number of stored identities."""
        conn = self._connect()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM persons")
        count = cur.fetchone()[0]

        conn.close()
        return count