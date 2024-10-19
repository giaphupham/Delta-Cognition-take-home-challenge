import cv2
from src.object_detection.yolo_detector import YOLODetector
from src.novelty_detection.autoencoder import Autoencoder
from src.collision_avoidance.kalman_filter import KalmanFilterCollisionAvoidance
from src.utils.video_utils import open_video, read_frame, display_frame, release_video

def main():
    # Initialize YOLO object detector
    yolo_detector = YOLODetector()

    # Initialize novelty detection (autoencoder for example)
    novelty_detector = Autoencoder(input_dim=784)

    # Initialize collision avoidance system
    collision_avoidance_system = KalmanFilterCollisionAvoidance()

    # Open the video file
    cap = open_video('data/test_videos/test_video.mp4')

    # Main loop to process video frames
    while True:
        frame = read_frame(cap)
        if frame is None:
            break

        # Object detection
        detections, frame_with_detections = yolo_detector.detect_objects(frame)

        # Display detections
        display_frame(frame_with_detections, 'Object Detection')

        # Collision avoidance logic (example)
        for detection in detections:
            object_position = detection[:2]  # Extract object x, y position
            vehicle_position = [320, 240]  # Example vehicle position (center of frame)
            if collision_avoidance_system.check_collision_risk(object_position, vehicle_position, 50):
                print("Collision risk detected! Taking evasive action...")

    release_video(cap)

if __name__ == '__main__':
    main()
