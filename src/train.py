from ultralytics import YOLO

# Load YOLOv8 model (nano version, optimized for CPU)
model = YOLO("yolov8n.pt")  # You can use "yolov8s.pt" for a slightly larger model

# Train the model on your dataset
model.train(
    data="ReLeaf1Dataset/data.yaml",  # Path to your dataset config file
    epochs=50,  # Number of training epochs
    imgsz=640,  # Image size for training
    batch=4,  # Adjust based on your CPU power
    device="cpu"  # Ensure it runs on CPU
)

# Save the trained model
model.export(format="torchscript")  # Saves the model in PyTorch format
