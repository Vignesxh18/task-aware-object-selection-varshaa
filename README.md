# Task Aware Object Selection System

## Project Overview

This project implements a **task-aware object selection framework** based on the COCO dataset.
Instead of detecting all objects in an image, the system selects the **most relevant object for a given task**.

Example:

Task: watch tv
Detected objects: tv, chair, person

Output: **tv**

The system combines **object detection and task understanding** to select the most suitable object.

---

# System Pipeline

Image + Task Prompt
↓
YOLO Object Detection
↓
Feature Extraction
↓
Task Prompt Encoding (CLIP)
↓
Similarity Calculation
↓
Best Object Selection

---

# Technologies Used

* Python
* YOLOv8 (Object Detection)
* CLIP (Task Understanding)
* COCO Dataset

---

# Example

Input Image: Living room

Detected Objects:

* tv
* chair
* person
* vase

Task:
watch tv

Output:
Selected Object → **tv**

---

# Hardware Architecture (VLSI Perspective)

The proposed system is designed for deployment on **edge hardware platforms**.

Architecture:

Camera / Image Input
↓
Preprocessing Unit
↓
YOLO CNN Accelerator
↓
Feature Extraction
↓
Task Encoder
↓
Similarity Engine
↓
Object Selector

The final system will be implemented on:

* **VEGA Processor**
* **Genesys-2 FPGA**

---

# Repository Structure

task_selector.py → Main pipeline
test_yolo.py → YOLO detection test
test_clip.py → CLIP embedding test
test_torch.py → Torch environment test
val2017/ → COCO dataset images
output.jpg → Example output

---

# How to Run

Install dependencies:

pip install ultralytics torch torchvision pillow openai-clip

Run the system:

python task_selector.py

Enter image filename and task when prompted.

Example:

000000000139.jpg
watch tv

Output:

Selected Object → tv

---

# Future Work

* FPGA acceleration
* YOLO hardware accelerator
* Task matching accelerator
* Deployment on VEGA processor
