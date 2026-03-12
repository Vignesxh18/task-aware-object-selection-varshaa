import os

import clip
import cv2
import torch
from PIL import Image
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"

# load models
clip_model, preprocess = clip.load("ViT-B/32", device=device)
yolo_model = YOLO("yolov8n.pt")

# input image
image_folder = "val2017"
image_file = input("Enter COCO image filename: ")

image_path = os.path.join(image_folder, image_file)

if not os.path.exists(image_path):
    print("Image not found")
    exit()

image = Image.open(image_path)
image_cv = cv2.imread(image_path)

print("\nUsing image:", image_file)

# task input
task_prompt = input("Enter task (example: serve wine, sit, read, drink): ")

print("\nTask:", task_prompt)

# -----------------------------
# Step 1 YOLO detection
# -----------------------------

results = yolo_model(image_path)

objects = []
boxes = []
scores = []

for r in results:
    for box in r.boxes:

        cls_id = int(box.cls)
        conf = float(box.conf)

        obj = yolo_model.names[cls_id]

        if conf > 0.30:

            objects.append(obj)
            boxes.append(box.xyxy[0])

print("\nDetected Objects:")

for o in objects:
    print(o)

# -----------------------------
# Step 2 task scoring
# -----------------------------

clip_scores = []

for obj,box in zip(objects,boxes):

    x1,y1,x2,y2 = map(int,box)

    crop = image.crop((x1,y1,x2,y2))

    image_input = preprocess(crop).unsqueeze(0).to(device)

    prompt = f"best object to {task_prompt}"

    text = clip.tokenize([prompt]).to(device)

    with torch.no_grad():

        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T

    score = similarity.cpu().numpy()[0][0]

    clip_scores.append(score)

print("\nTask Relevance Scores:")

for o,s in zip(objects,clip_scores):
    print(o,"→",round(s,3))

# -----------------------------
# Step 3 select best object
# -----------------------------

best_index = clip_scores.index(max(clip_scores))

selected_object = objects[best_index]
best_box = boxes[best_index]

print("\nFINAL RESULT")
print("Task:",task_prompt)
print("Selected Object:",selected_object)

# -----------------------------
# Step 4 visualize result
# -----------------------------

x1,y1,x2,y2 = map(int,best_box)

cv2.rectangle(image_cv,(x1,y1),(x2,y2),(0,255,0),3)

label = f"{task_prompt} → {selected_object}"

cv2.putText(image_cv,label,(x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,(0,255,0),2)

cv2.imwrite("output.jpg",image_cv)

print("\nSaved output image as output.jpg")