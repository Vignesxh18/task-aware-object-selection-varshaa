import torch
from transformers import CLIPModel, CLIPProcessor

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

task = "person riding skateboard"

inputs = processor(text=task, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    text_outputs = model.text_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

    pooled_output = text_outputs.pooler_output
    text_features = model.text_projection(pooled_output)

print("Text embedding shape:", text_features.shape)