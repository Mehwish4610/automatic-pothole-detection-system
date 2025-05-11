
from roboflow import Roboflow
rf = Roboflow(api_key="eU14swkqXttRwj0yLOEf")
project = rf.workspace("potholedetection-yntcc").project("pothole-detection-system-y79un")
version = project.version(3)
dataset = version.download("yolov8")
                               