# Get_my_YOLO_on.py
# Get_my_YOLO_on.py — now with full-image view!

from ultralytics import YOLO
import cv2
from pathlib import Path

# Load the nano model (fast on CPU)
model = YOLO("yolov8n.pt")

# Your image
image_path = Path(r"K:\Year 2025\2025-12-06 Jefferson Forest vs Varina Football\5Q2A0690.JPG")

img = cv2.imread(str(image_path))
if img is None:
    print("Failed to load image!")
    exit()

# Run YOLO inference
results = model(img)

# Print detections
for r in results:
    print(f"Found {len(r.boxes)} objects")
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        print(f"  {label} ({conf:.2f}): bbox=({x1}, {y1}, {x2}, {y2})")

# Draw boxes on the original image
annotated = results[0].plot()  # YOLO draws labels + boxes

# Resize for full-screen viewing (keeps aspect ratio, max 1600 px long side)
max_display_size = 1600
h, w = annotated.shape[:2]
scale = min(max_display_size / w, max_display_size / h)
new_w = int(w * scale)
new_h = int(h * scale)

resized = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Show it!
cv2.imshow(f"YOLO Detection - {image_path.name}", resized)
print(f"\nDisplaying resized image: {new_w}x{new_h} (original was {w}x{h})")
cv2.waitKey(0)
cv2.destroyAllWindows()