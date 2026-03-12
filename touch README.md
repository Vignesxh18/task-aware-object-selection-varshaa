# Task-Aware Object Selection Framework  
DVCON India 2026 – Design Contest Submission  
Author: Vignesh S  

---

## Problem Statement

Conventional object detection systems detect all objects in a scene without understanding which object is relevant to a given task.

This project implements a Task-Aware Object Selection Framework that:

- Takes an image  
- Takes a task description (text prompt)  
- Selects the most relevant object for that task  

---

## Proposed Approach

The system combines:

- YOLOv8n for object detection  
- CLIP (ViT-B/32) for cross-modal text-image embeddings  
- Cosine similarity for task-object relevance scoring  

---

## System Pipeline

1. Input image  
2. YOLO detects objects  
3. Each object is cropped  
4. CLIP generates:
   - Text embedding (task prompt)
   - Image embedding (object crop)
5. Cosine similarity is computed  
6. The object with highest similarity is selected  

---

## Architecture Overview

Image  
↓  
YOLOv8n (Object Detection)  
↓  
Object Crops  
↓  
CLIP Vision Encoder  
↓  
512-D Image Embedding  

Task Prompt  
↓  
CLIP Text Encoder  
↓  
512-D Text Embedding  

Cosine Similarity  
↓  
Best Matching Object  

---

## Example

Task:
```
drive car
```

Detected Objects:
- person  
- car  
- car  
- car  
- skateboard  

Similarity Scores:
```
person → 0.1976  
car → 0.2268 (highest)  
skateboard → 0.1975  
```

Selected Object:
car  

---

## Implementation Details

- Python 3.11  
- PyTorch  
- Ultralytics YOLOv8n  
- HuggingFace Transformers (CLIP)  
- OpenCV  

---

## Project Structure

```
task_aware_project/
│
├── task_selector.py      # Main pipeline
├── test_yolo.py          # YOLO test
├── test_clip.py          # CLIP test
├── sample.jpg            # Test image
├── output.jpg            # Result image
└── README.md
```

---

## Hardware Acceleration Plan (Stage 2)

For FPGA deployment:

- Quantized YOLO backbone (INT8)  
- Fixed-point cosine similarity accelerator  
- CLIP text embedding precomputed offline  
- VEGA processor handles control flow  
- FPGA accelerates convolution-heavy layers  

---

## Key Contribution

- Task-driven object prioritization  
- Cross-modal semantic matching  
- Edge-compatible architecture  
- FPGA acceleration-ready design  

---

## Future Work

- ONNX export  
- Model quantization  
- FPGA mapping  
- VEGA integration  
- Latency optimization  

---

## License

For academic and contest use.