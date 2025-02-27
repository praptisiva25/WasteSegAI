from ultralytics import YOLO

# Load the trained model
model = YOLO("C:/Users/prapt/OneDrive/Desktop/Project/WasteSegAI/runs/detect/train/weights/best.pt")  # Path to your best model

# Run inference on an image
results = model.predict("test_image2.jpg", save=True, save_dir="runs/detect/custom_predict")

# Print results (optional)
for r in results:
    print(r.boxes)  # Bounding box coordinates
    print(r.boxes.conf)  # Confidence scores
 