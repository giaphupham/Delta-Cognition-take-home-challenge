import cv2

def resize_image(image, size=(640, 480)):
    # Resize image to a fixed size
    return cv2.resize(image, size)

def normalize_image(image):
    # Normalize image pixel values between 0 and 1
    return image / 255.0

def preprocess_image(image, size=(640, 480)):
    # Resize and normalize the image
    resized = resize_image(image, size)
    normalized = normalize_image(resized)
    return normalized
