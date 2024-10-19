import numpy as np
import cv2

class KalmanFilterCollisionAvoidance:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def predict(self, object_position):
        # Update measurement with the object's current position
        self.kalman.correct(np.array([[np.float32(object_position[0])], [np.float32(object_position[1])]]))
        # Predict the future position
        predicted = self.kalman.predict()
        return predicted

    def check_collision_risk(self, object_position, vehicle_position, safe_distance):
        predicted_position = self.predict(object_position)
        distance = np.linalg.norm(predicted_position[:2] - np.array(vehicle_position))
        return distance < safe_distance
