from ultralytics import YOLO

model = YOLO("C:/Users/prapt/OneDrive/Desktop/Project/WasteSegAI/runs/detect/train/weights/best.pt")  # Replace with your model file
print(model.names)  # This prints the class ID-to-name mapping
