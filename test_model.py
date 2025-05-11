from ultralytics import YOLO

model = YOLO(r"E:\potholeDetector\runs\segment\pothole-seg2\weights\best.pt")
results = model.predict(source=r"", 
                        imgsz=1024,
                        save=True,   # 👈 this saves the output image
                        conf=0.3,
                        iou=0.4)
