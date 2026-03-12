import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("sample.jpg")).unsqueeze(0).to(device)

texts = [
    "a person walking",
    "a car on the road",
    "a person riding skateboard"
]

text_tokens = clip.tokenize(texts).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)

similarity = (image_features @ text_features.T).softmax(dim=-1)

scores = similarity.cpu().numpy()[0]

print("\nCLIP Similarity Scores:")

for t, s in zip(texts, scores):
    print(f"{t} → {s:.3f}")