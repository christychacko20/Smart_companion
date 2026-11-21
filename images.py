import os
import cv2
import numpy as np
import cv2
import dlib
import time
from scipy.spatial import distance as dist
from imutils import face_utils
import csv

face_model = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def analyze_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_model(gray)

    if len(faces) == 0:
        print("No face detected in image.")
        return

    for face in faces:
        shapes = landmark_model(gray, face)
        shape = face_utils.shape_to_np(shapes)

        lip_dist = cal_yawn(shape)
        print("\n---- [IMAGE] Parameters regarding mouth ----")
        print("Distance between lips: ", lip_dist)

        eye_dist = cal_eye(shape)
        ear_avg = (eye_dist[2] + eye_dist[3]) / 2
        print("\n---- [IMAGE] Parameters regarding eyes ----")
        print(f"EAR Left: {eye_dist[2]}, EAR Right: {eye_dist[3]}, EAR Avg: {ear_avg}")

        # Detect Eye State
        if ear_avg < 0.25:
            print("Eyes appear CLOSED in image.")
        else:
            print("Eyes appear OPEN in image.")

        # Detect Yawning
        if lip_dist > yawn_thresh:
            print("YAWNING detected in image.")
        else:
            print("NO YAWNING detected in image.")

        # Optionally visualize
        cv2.putText(image, f"EAR: {ear_avg:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Analyzed Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
analyze_image("eye_closed2.jpeg")  # Replace with path to your image
