import cv2

def open_video(video_path):
    # Open a video file or camera stream
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Unable to open video.")
    return cap

def read_frame(cap):
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def release_video(cap):
    # Release the video capture object
    cap.release()

def display_frame(frame, window_name='Frame'):
    # Display a single frame
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)
