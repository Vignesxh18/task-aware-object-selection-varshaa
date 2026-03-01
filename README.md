---

## Setup and Installation Guide

Follow these steps to run the Task-Aware Object Selection pipeline locally.

---

### 1. Clone the Repository

```bash
git clone https://github.com/Vigneshx18/task-aware-object-selection.git
cd task-aware-object-selection
```

---

### 2. Create Virtual Environment (Recommended)

Make sure Python 3.11 is installed.

```bash
python3.11 -m venv venv
source venv/bin/activate     # macOS / Linux
```

On Windows:
```bash
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install ultralytics
pip install transformers
pip install opencv-python
```

---

### 4. Add a Test Image

Place any image inside the project folder and rename it:

```
sample.jpg
```

Example structure:

```
task-aware-object-selection/
│
├── task_selector.py
├── sample.jpg
├── README.md
└── venv/
```

---

### 5. Run the Task-Aware Selection Pipeline

```bash
python task_selector.py
```

You will be prompted:

```
Enter task:
```

Example input:
```
drive car
```

---

### 6. Output

The program will:

- Detect objects using YOLOv8
- Compute semantic similarity using CLIP
- Select the most relevant object
- Save result as:

```
output.jpg
```

---

## Expected Example Output

Detected Objects:
- person
- car
- skateboard

Similarity Scores:
```
person → 0.19
car → 0.22
skateboard → 0.19
```

Selected Object:
car

---

## Notes

- The first run will download model weights automatically.
- Internet connection is required during first execution.
- For best results, use clear images containing COCO dataset objects.
