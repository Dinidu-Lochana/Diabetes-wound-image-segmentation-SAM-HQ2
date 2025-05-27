import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

# Set the path to include the GroundingDINO module
sys.path.append(os.path.join(os.path.dirname(__file__), "GroundingDINO"))

from groundingdino.util.inference import load_model, predict
from torchvision import transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"
model = load_model(CONFIG_PATH, WEIGHTS_PATH)
model.to(device)

# Load and transform the image
image_path = "test_10/test1.jpg"  # 🔁 Replace with your image
image_pil = Image.open(image_path).convert("RGB")

# Transform image to tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image_pil).to(device)  # ✅ No .unsqueeze(0)

# Define the text prompt
prompt = "diabetic wound"

# Perform prediction
boxes, logits, phrases = predict(
    model=model,
    image=image_tensor,  # ✅ Pass tensor directly, not [None]
    caption=prompt,
    box_threshold=0.3,
    text_threshold=0.25,
    device=device
)


# Convert boxes to numpy for visualization
boxes = boxes.cpu().numpy()

# Show image and draw bounding boxes
fig, ax = plt.subplots(1)
ax.imshow(image_pil)

for box, phrase in zip(boxes, phrases):
    x0, y0, x1, y1 = box
    print(f"Detected '{phrase}' at: ({x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f})")
    rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x0, y0, phrase, color='white', backgroundcolor='red', fontsize=12)

plt.axis("off")
plt.tight_layout()
plt.show()
