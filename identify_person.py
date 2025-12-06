# identify_person.py
from deepface import DeepFace
import numpy as np
import os
from database import FaceDatabase

# Force CPU for DeepFace on Mac (optional, keeps things stable)
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class FaceIdentifier:
    def __init__(self, db: FaceDatabase, threshold: float = 0.4, alpha: float = 0.1):
        """
        threshold: max L2 distance to consider same person
        alpha: EMA blending factor for updating canonical embeddings
        """
        self.db = db
        self.threshold = threshold
        self.alpha = alpha

    # ------------------------------------------------------
    # 1. EMBEDDING (ArcFace → L2 normalized)
    # ------------------------------------------------------
    def get_embedding(self, face_image):
        """
        face_image: cropped BGR OpenCV face
        Returns: L2-normalized embedding vector
        """
        rep = DeepFace.represent(
            img_path=face_image,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="skip"
        )

        emb = np.array(rep[0]["embedding"], dtype=np.float32)

        # L2 normalize
        norm = np.linalg.norm(emb) + 1e-10
        emb = emb / norm
        return emb

    # ------------------------------------------------------
    # 2. MATCH AGAINST DATABASE
    # ------------------------------------------------------
    def match_face(self, embedding):
        persons = self.db.get_all_persons()  # returns dict: {id: embedding}

        if not persons:
            return None, 1e9

        best_id = None
        best_dist = 1e9

        for pid, stored_emb in persons.items():
            dist = np.linalg.norm(embedding - stored_emb)
            if dist < best_dist:
                best_dist = dist
                best_id = pid

        return best_id, best_dist

    # ------------------------------------------------------
    # 3. IDENTIFY OR CREATE NEW PERSON
    # ------------------------------------------------------
    def identify_or_create(self, embedding, timestamp: str = ""):
        """
        embedding: L2-normalized embedding vector
        timestamp: optional for logging
        Returns:
            (person_id, distance, is_new)
        """
        best_id, best_dist = self.match_face(embedding)

        # --------------------------------------------------
        # CASE 1: Recognized person (distance < threshold)
        # --------------------------------------------------
        if best_id is not None and best_dist < self.threshold:

            # Get the OLD embedding
            persons = self.db.get_all_persons()
            old_emb = persons.get(best_id)

            # EMA update for canonical embedding
            new_emb = (1.0 - self.alpha) * old_emb + self.alpha * embedding
            new_emb /= (np.linalg.norm(new_emb) + 1e-10)

            self.db.update_person_embedding(best_id, new_emb)
            self.db.add_face_observation(best_id, embedding, timestamp)

            return best_id, best_dist, False  # not a new identity

        # --------------------------------------------------
        # CASE 2: New person
        # --------------------------------------------------
        new_id = self.db.add_person(embedding)
        self.db.add_face_observation(new_id, embedding, timestamp)

        return new_id, best_dist, True  # new identity