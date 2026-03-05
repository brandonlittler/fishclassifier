import torch
from PIL import Image
from model import FishClassifier
from transforms import build_transforms
import json
"""
Random test script that can be used to test on an unseen (train/val) image. 
Can just drop an image into the workdir fol for now... 
"""



# Load the model
model = FishClassifier(num_classes=23)
model.load_state_dict(torch.load("fish_classifier.pth", map_location="cpu"))
model.eval()  # Set to evaluation mode

# Get transforms (use validation transforms - no augmentation)
_, val_transforms = build_transforms(img_size=128)

# load and pre-process the image
image_path = "/Users/rhyslittler/clean-fish-classification/test002.jpeg"
img = Image.open(image_path).convert("RGB")
img_tensor = val_transforms(img).unsqueeze(0)  

# Make prediction
with torch.no_grad():  
    outputs = model(img_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    predicted_class = outputs.argmax(dim=1).item()
    confidence = probabilities[0][predicted_class].item()

     # show the top 3 predictions
    top3_probs, top3_indices = torch.topk(probabilities, 3)


with open("labels.json", "r") as f:
    labels = json.load(f)
# Map class index to fish name
fish_name = labels[str(predicted_class)]

print(f"Predicted class: {predicted_class}")
print(f"Fish species: {fish_name}")
print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

print(f"Image size: {img.size}")
print(f"Tensor shape: {img_tensor.shape}")
print(f"Tensor min/max: {img_tensor.min():.4f}, {img_tensor.max():.4f}")

print("\nTop 3 predictions:")
# loop through the top 3 predictions and print the label and probability
for i in range(3):
        idx = top3_indices[0][i].item()
        prob = top3_probs[0][i].item()
        print(f"  {i+1}. {labels[str(idx)]}: {prob:.4f} ({prob*100:.2f}%)")