import cv2
import os

# Load the image
image_path = "test_10/test1.jpg"
image = cv2.imread(image_path)

# Check if image was loaded successfully
if image is None:
    raise FileNotFoundError(f"Could not load image. Check the path: {image_path}")

# Bounding box coordinates (x_min, y_min, x_max, y_max)
bounding_boxes = [
    (2, 0, 564, 564)
]

# Draw bounding boxes
for (x_min, y_min, x_max, y_max) in bounding_boxes:
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)  # Red box

# Create output directory if it doesn't exist
output_dir = "outputs_testing/outputs_testing_from_Peplexity"
os.makedirs(output_dir, exist_ok=True)

# Save the new image
output_path = os.path.join(output_dir, "wounds_bounding_box1.jpg")
success = cv2.imwrite(output_path, image)

# Confirm save
if success:
    print(f"✅ Image saved successfully to: {output_path}")
else:
    print("❌ Failed to save image.")
