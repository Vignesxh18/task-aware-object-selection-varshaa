import cv2
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load models
yolo = YOLO("yolov8n.pt")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_text_embedding(task):
    inputs = processor(text=task, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_outputs = clip_model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        pooled = text_outputs.pooler_output
        text_features = clip_model.text_projection(pooled)

    return text_features / text_features.norm(dim=-1, keepdim=True)

def get_image_embedding(image_crop):
    image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        vision_outputs = clip_model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        pooled_output = vision_outputs.pooler_output
        image_features = clip_model.visual_projection(pooled_output)

    # normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features

def main():
    image_path = "sample.jpg"
    task = input("Enter task: ")

    text_embedding = get_text_embedding(task)

    results = yolo(image_path)
    image = cv2.imread(image_path)

    best_score = -1
    best_label = None
    best_box = None

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = yolo.names[int(box.cls)]

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        image_embedding = get_image_embedding(crop)
        similarity = torch.cosine_similarity(text_embedding, image_embedding)
        score = similarity.item()

        print(f"{label} -> {score:.4f}")

        if score > best_score:
            best_score = score
            best_label = label
            best_box = (x1, y1, x2, y2)

    print("\nSelected Object:", best_label)
    print("Similarity Score:", best_score)

    # Draw bounding box
    if best_box:
        x1, y1, x2, y2 = best_box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(image, best_label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

        cv2.imwrite("output.jpg", image)
        print("Saved result as output.jpg")

if __name__ == "__main__":
    main()