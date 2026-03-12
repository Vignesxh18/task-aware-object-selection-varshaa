from ultralytics import YOLO

# load model
model = YOLO("yolov8n.pt")

# run detection
results = model("sample.jpg")

print("\nDetected Objects:")

for r in results:
    for c in r.boxes.cls:
        obj = model.names[int(c)]
        print(obj)