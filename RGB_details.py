import cv2
import matplotlib.pyplot as plt

# Load image using OpenCV (reads in BGR format)
image_path = "inputs/test1.jpg"
image = cv2.imread(image_path)

# Check if image was loaded
if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Split RGB channels
R, G, B = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]

# Plot original image and channels
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(R, cmap="Reds")
plt.title("Red Channel")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(G, cmap="Greens")
plt.title("Green Channel")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(B, cmap="Blues")
plt.title("Blue Channel")
plt.axis("off")

plt.tight_layout()
plt.show()
