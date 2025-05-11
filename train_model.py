from ultralytics import YOLO

# Load base YOLOv8 segmentation model
model = YOLO("yolov8s-seg.pt")  # You can use yolov8m-seg.pt for better accuracy

# Train on custom Roboflow dataset
model.train(data='E:\potholeDetector\pothole-Detection-System-3\data.yaml', epochs=50, imgsz=640, conf = 0.3, iou = 0.4)
