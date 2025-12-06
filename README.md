# SentinelVision  
### Real-Time Weapon Detection + Face Recognition System  
**Author:** Connor Reyes  

SentinelVision is an end-to-end real-time computer vision system designed to identify faces, detect weapons, assign threat levels, and persist identities across sessions.  
It uses **YOLOv8**, **DeepFace embeddings**, **OpenCV**, and a **SQLite identity database** to create a production-style AI surveillance pipeline.

---

## Features

### Face Detection
- YOLOv8 fast, lightweight face detection  
- Multi-face tracking  
- Cropped faces fed directly to DeepFace

### Identity Recognition
- ArcFace/SFace embeddings (L2-normalized)  
- Accurate cosine-similarity matching  
- Auto-creation of new identities  
- Adaptive EMA embedding refinement  
- Persistent storage in SQLite for long-term recognition  

### Weapon Detection
- Custom YOLOv8s model trained on **14,000+ weapon images**  
- Detects pistols, rifles, knives  
- Real-time bounding boxes & confidence scores  

### Threat Assessment
- HIGH THREAT if weapon detected  
- LOW when no weapon detected  
- Color-coded real-time overlays  

### High-FPS Pipeline
- Threaded webcam video capture  
- DeepFace called only on spaced intervals  
- Embedding caching to reduce recomputation  

---

## System Architecture
flowchart TD

%% ==== INPUT LAYER ====
A[📷 Camera Input] --> B[Threaded Webcam Capture\n(WebcamStream)]

%% ==== FACE DETECTION ====
B --> C[YOLOv8n-Face\nFace Detector]

C -->|Face Bounding Boxes| D[Crop Face Regions]

%% ==== EMBEDDING PIPELINE ====
D --> E[DeepFace ArcFace\n(L2-normalized Embedding)]

E --> F[Identify_or_Create()\nFaceIdentifier]

%% ==== IDENTITY MATCHING ====
F -->|Match < Threshold| G[Existing Person\n(Update with EMA)]
F -->|No Match| H[Create New Person]

G --> I[Update Person Embedding\n(database)]
H --> J[Insert New Person\n(database)]

%% ==== SIGHTING LOG ====
F --> K[Add Face Observation\n(timestamped)]

%% ==== WEAPON DETECTION PARALLEL ====
B --> L[YOLOv8s-Weapon\nDetector]
L --> M{Weapon Detected?}

M -->|Yes| N[THREAT_LEVEL = HIGH]
M -->|No| O[THREAT_LEVEL = LOW]

%% ==== OVERLAY + OUTPUT ====
D --> P[Draw Face Boxes + IDs]
M --> P
P --> Q[Display Final Frame\n(OpenCV imshow)]

%% ==== DATABASE ====
I --> R[(SQLite: persons)]
J --> R
K --> S[(SQLite: faces)]

style A fill:#e3f2fd,stroke:#42a5f5,stroke-width:2px
style C fill:#fff3cd,stroke:#ffca2c,stroke-width:2px
style D fill:#fff3cd,stroke:#ffca2c,stroke-width:2px
style E fill:#e8f5e9,stroke:#66bb6a,stroke-width:2px
style F fill:#e8f5e9,stroke:#66bb6a,stroke-width:2px
style G fill:#c8e6c9,stroke:#43a047,stroke-width:2px
style H fill:#c8e6c9,stroke:#43a047,stroke-width:2px
style L fill:#fdecea,stroke:#f44336,stroke-width:2px
style M fill:#ffebee,stroke:#e53935,stroke-width:2px
style P fill:#ede7f6,stroke:#7e57c2,stroke-width:2px
style Q fill:#ede7f6,stroke:#7e57c2,stroke-width:2px

## 📦 Tech Stack

- **Python 3.9**
- **YOLOv8 (Ultralytics)**
- **DeepFace (ArcFace/SFace)**
- **OpenCV**
- **NumPy**
- **SQLite3**
- **Google Colab / A100 GPU for training**

---

## 🔧 Installation

### 1. Clone the repository
``bash
git clone https://github.com/connoreyes/sentinelvision.git
cd sentinelvision

### 2. Create enviorment
conda create -n sv python=3.9
conda activate sv

### 3. Install dependencies
pip install ultralytics deepface opencv-python numpy

### 4. Run SentinelVision
python main.py

---

## Identity Tracking Logic
Step 1 — Face detected

YOLO outputs coordinates, the frame is cropped.

Step 2 — Embedding generated

DeepFace (ArcFace/SFace) generates a 512-dim normalized vector.

Step 3 — Compare to known identities

Cosine similarity or L2 distance is computed.

If similarity > threshold → same person
Else → new person saved to DB

Step 4 — Embedding refinement

To stabilize identity:
new_emb = 0.9 * old_emb + 0.1 * new_emb

---

 Weapon Detection Model

Trained on:
	•	14,000+ weapon images
	•	Pistol, rifle, knife categories
	•	Google Colab PRO (NVIDIA A100)
	•	Training used augmentation & YOLOv8s architecture

Achieved metrics:
	•	mAP50 ≈ 0.64
	•	Strong real-world detection

Your model is located at:
models/yolov8s-weapon.pt

### Project Structure
sentinelvision/
│── main.py
│── detect_face.py
│── identify_person.py
│── threat_logic.py
│── database.py
│── models/
│    ├── yolov8n-face.pt
│    ├── yolov8s-weapon.pt
│── data/
│    └── embeddings.db
│── README.md

### Future Improvements
	•	Multi-camera support
	•	Person re-identification across angles
	•	Body pose aggression modeling
	•	Jetson Nano / Raspberry Pi edge inference
	•	Cloud dashboard for identity logs
	•	Multiclass weapon training
	•	GPU acceleration for Mac M-series

### Author

Connor Reyes
Software Engineer | AI/ML Developer
	•	Email: connorreyes05@gmail.com
	•	GitHub: https://github.com/connoreyes
	•	LinkedIn: https://www.linkedin.com/in/connor-reyes-4b33932a9/
