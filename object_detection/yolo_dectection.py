import torch
import cv2

class YOLODetector:
    def __init__(self, model_path='yolov5s'):
        # Load pre-trained YOLO model from ultralytics repository
        self.model = torch.hub.load('ultralytics/yolov5', model_path, pretrained=True)

    def detect_objects(self, frame):
        # Perform inference on the input frame
        results = self.model(frame)
        # Return the detections and the frame with bounding boxes rendered
        return results.xyxy[0], results.render()[0]

    def show_frame(self, frame):
        cv2.imshow('YOLO Object Detection', frame)
        cv2.waitKey(1)
