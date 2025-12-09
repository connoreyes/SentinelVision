# Import DeepFace for face embeddings (ArcFace, Facenet, etc.)
from deepface import DeepFace

# NumPy for vector math on embeddings
import numpy as np

# OS used here only to modify environment variables (disable GPU)
import os

# Import our custom database class (stores persons + observations)
from database import FaceDatabase


# ---------------------------------------------------------------
# Disable GPU for DeepFace
# ---------------------------------------------------------------
# On MacBooks, TensorFlow GPU is unstable and can cause crashes.
# Setting CUDA_VISIBLE_DEVICES to "" forces DeepFace/TensorFlow to run on CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ===================================================================
# FACE IDENTIFIER CLASS
# ===================================================================
class FaceIdentifier:
    def __init__(self, db: FaceDatabase, threshold: float = 0.4, alpha: float = 0.1):
        """
        db        = our database instance for storing/loading embeddings
        threshold = maximum allowed L2 distance to consider a 'match'
        alpha     = smoothing factor for EMA updating of canonical embeddings

        WHY THRESHOLD?
        - Smaller threshold = strict matching (fewer false positives)
        - Larger threshold = looser matching (but risk of mixing identities)

        WHY ALPHA?
        - As you see a person's face under slightly different angles/lighting,
          you want the canonical embedding to slowly adapt → EMA (exponential moving average).
        """
        self.db = db
        self.threshold = threshold
        self.alpha = alpha


    # ===================================================================
    # 1. EMBEDDING EXTRACTION (DeepFace → ArcFace)
    # ===================================================================
    def get_embedding(self, face_image):
        """
        face_image: cropped face image from your detector, BGR format
        Returns: a L2-normalized embedding vector (np.ndarray)

        WHY L2 NORMALIZE?
        - ArcFace embeddings are meant to be compared using cosine distance.
        - Normalizing ensures consistent scale and improves matching reliability.
        """

        rep = DeepFace.represent(
            img_path=face_image,       # can pass a NumPy array directly
            model_name="ArcFace",      # best for identity recognition
            enforce_detection=False,   # we already have the face cropped → skip DF detector
            detector_backend="skip"    # skip unnecessary face detection step
        )

        # Convert the embedding list to NumPy vector
        emb = np.array(rep[0]["embedding"], dtype=np.float32)

        # Compute magnitude (L2 norm) and normalize vector
        norm = np.linalg.norm(emb) + 1e-10   # epsilon avoids divide-by-zero
        emb = emb / norm

        return emb


    # ===================================================================
    # 2. MATCH AGAINST EXISTING IDENTITIES (L2 DISTANCE)
    # ===================================================================
    def match_face(self, embedding):
        """
        Compare a new embedding to all stored embeddings.
        Returns:
            (best_id, best_distance)

        If database is empty → returns (None, huge_number)
        """

        persons = self.db.get_all_persons()  # { person_id : embedding }

        if not persons:
            return None, 1e9                 # No identities yet

        best_id = None
        best_dist = 1e9                      # large placeholder value

        # Loop through every stored identity's canonical embedding
        for pid, stored_emb in persons.items():

            # L2 distance between new embedding and stored embedding
            dist = np.linalg.norm(embedding - stored_emb)

            # Save best match if closer than previous
            if dist < best_dist:
                best_dist = dist
                best_id = pid

        return best_id, best_dist


    # ===================================================================
    # 3. IDENTIFY OR CREATE NEW IDENTITY
    # ===================================================================
    def identify_or_create(self, embedding, timestamp: str = ""):
        """
        Main function used by SentinelVision.

        embedding:  L2-normalized embedding from ArcFace
        timestamp:  optional - logs when identity was seen

        RETURNS:
            (person_id, distance, is_new)

        Behavior:
        - If embedding matches an existing person → refine that person's stored embedding
        - Else → create a new person in the database
        """

        best_id, best_dist = self.match_face(embedding)

        # --------------------------------------------------------------
        # CASE 1: Existing identity recognized
        # --------------------------------------------------------------
        if best_id is not None and best_dist < self.threshold:

            # Get all persons again (dictionary)
            persons = self.db.get_all_persons()

            # Retrieve the old canonical embedding for this person
            old_emb = persons.get(best_id)

            # ----------------------------------------------------------
            # EMA UPDATE (EXPERIMENTAL BUT IMPORTANT)
            # ----------------------------------------------------------
            # New canonical embedding:
            #     new = (1 - alpha)*old + alpha*new
            #
            # WHY?
            # - A person’s appearance changes slightly due to lighting, angle,
            #   distance, facial expressions, etc.
            # - Instead of storing *only the first embedding*, we refine it over time.
            #
            new_emb = (1.0 - self.alpha) * old_emb + self.alpha * embedding

            # Normalize updated embedding again (L2 normalize)
            new_emb /= (np.linalg.norm(new_emb) + 1e-10)

            # Save updated embedding back to database
            self.db.update_person_embedding(best_id, new_emb)

            # Log the face observation (helps with debugging and analytics)
            self.db.add_face_observation(best_id, embedding, timestamp)

            # Not a new identity
            return best_id, best_dist, False


        # --------------------------------------------------------------
        # CASE 2: Identity NOT recognized → create new entry
        # --------------------------------------------------------------
        new_id = self.db.add_person(embedding)

        # Log first sighting in observation table
        self.db.add_face_observation(new_id, embedding, timestamp)

        # Return the new person's ID
        return new_id, best_dist, True