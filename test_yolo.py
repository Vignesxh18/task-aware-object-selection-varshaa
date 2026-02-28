from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("sample.jpg")
results[0].show()
