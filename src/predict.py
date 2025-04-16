from ultralytics import YOLO


model = YOLO("C:/Users/prapt/OneDrive/Desktop/Project/WasteSegAI/runs/detect/train/weights/best.pt")  


results = model.predict("C:/Users/prapt/Favorites/projects/WasteSegAI/test_image3.jpg", save=True, save_dir="runs/detect/custom_predict")


for r in results:
    print(r.boxes)  
    print(r.boxes.conf)  
 