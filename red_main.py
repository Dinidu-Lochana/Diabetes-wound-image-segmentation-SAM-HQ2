import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from hydra import initialize_config_dir

# Add the SAM-HQ2 directory to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "sam_hq", "sam_hq2"))

# Initialize Hydra config
config_dir = os.path.join(
    os.path.dirname(__file__), 
    "sam_hq", "sam_hq2", "sam2", "configs"
)
initialize_config_dir(config_dir=config_dir, version_base="1.3")

# Import SAM-HQ2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Import YOLOv8
from ultralytics import YOLO

# -------------------------- Load Image with Red Channel --------------------------
image_path = "inputs/test18.jpg"  # Replace this with your image upload logic if needed
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError(f"❌ Image not found at: {image_path}")

# Extract only the red channel
red_channel = image_bgr[:, :, 2]  # OpenCV uses BGR, so index 2 is red

# Convert single channel to 3-channel by repeating red channel
image_r_rgb = cv2.merge([red_channel, red_channel, red_channel])  # fake RGB using red

# This is now used in YOLO and SAM pipeline
image_rgb = image_r_rgb


# -------------------------- Run YOLOv8 --------------------------
yolo_model = YOLO("runs/detect/train20/weights/best.pt")
results = yolo_model(image_path)[0]  # Get first result

# Extract original boxes and prompt points
original_boxes = []
prompt_points = []

for box in results.boxes.xyxy.cpu().numpy():
    x1, y1, x2, y2 = box
    original_boxes.append([x1, y1, x2, y2])

    # Get center point
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    prompt_points.append([center_x, center_y])

if not original_boxes:
    print("⚠️ No objects detected by YOLO. Exiting.")
    exit()

# -------------------------- Load SAM-HQ2 --------------------------
checkpoint_path = "sam_hq/sam_hq2/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "sam2.1/sam2.1_hiera_b+.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam_model = build_sam2(model_cfg, checkpoint_path)
sam_model.to(device)

predictor = SAM2ImagePredictor(sam_model)
predictor.set_image(image_rgb)

# -------------------------- Predict Masks --------------------------
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)

for i in range(len(original_boxes)):
    input_box = np.array([original_boxes[i]])         # shape (1, 4)
    input_point = np.array([[prompt_points[i]]])      # shape (1, 1, 2)
    input_label = np.array([[1]])                     # shape (1, 1)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        # box=input_box,
        multimask_output=False
    )

    x1, y1, x2, y2 = map(int, original_boxes[i])
    for mask in masks:
        # Clip mask to bounding box
        clipped_mask = np.zeros_like(mask, dtype=bool)
        clipped_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

        # Create RGBA overlay with yellow color and transparency
        yellow_mask = np.zeros((*clipped_mask.shape, 4), dtype=np.float32)
        yellow_mask[..., 0] = 1.0  # Red channel
        yellow_mask[..., 1] = 1.0  # Green channel
        yellow_mask[..., 2] = 0.0  # Blue channel
        yellow_mask[..., 3] = clipped_mask * 0.5  # Alpha channel

        # Mask outside pixels transparent
        masked_display = np.ma.masked_where(~clipped_mask, yellow_mask[..., 3])

        # Plot yellow mask overlay
        plt.imshow(yellow_mask)

    # Draw bounding box
    plt.gca().add_patch(plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        edgecolor='lime',
        facecolor='none',
        linewidth=2
    ))

# -------------------------- Save & Show --------------------------
plt.axis('off')
os.makedirs("outputs_SAM_HQ2_cropped_image", exist_ok=True)

# Auto-increment output filename
output_dir = "outputs_SAM_HQ2_cropped_image"
os.makedirs(output_dir, exist_ok=True)

# Find next available file name
i = 1
while True:
    output_path = os.path.join(output_dir, f"segmented{i}_with_boxes.png")
    if not os.path.exists(output_path):
        break
    i += 1

plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
print(f"✅ Segmented result saved to: {output_path}")

plt.show()
