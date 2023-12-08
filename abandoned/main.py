import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker  # Import the Tracker class from tracker.py
import math

model = YOLO('yolov8s.pt')

area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]
area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]

cap = cv2.VideoCapture('test.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

classes_to_track = ['person', 'handbag', 'suitcase']  # Define the classes to track

tracker = Tracker()  # Initialize the object tracker

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    objects_to_track = []  # Store the bounding boxes of tracked objects

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if c in classes_to_track:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green bounding boxes

            # Store bounding boxes of tracked objects along with their class labels
            objects_to_track.append((x1, y1, x2, y2, c))

    # Update the tracker with the bounding boxes of detected objects
    tracked_objects = tracker.update([obj[:4] for obj in objects_to_track])

    # Draw different colored bounding boxes for tracked objects
    colors = {'person': (255, 0, 0), 'handbag': (0, 255, 255), 'suitcase': (255, 255, 0)}
    for obj in tracked_objects:
        if len(obj) >= 5:  # Check if the sublist has at least 5 elements
            x1, y1, x2, y2, obj_id = obj
            if obj_id < len(objects_to_track) and len(objects_to_track[obj_id]) >= 5:  # Check if obj_id is valid
                class_label = objects_to_track[obj_id][4]
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors.get(class_label, (0, 0, 255)), 2)  # Use default color if class_label is not found in colors dictionary

    # Calculate distances between centers of different objects
    for i in range(len(tracked_objects)):
        for j in range(i + 1, len(tracked_objects)):
            x1, y1, _, _, _ = tracked_objects[i]
            x2, y2, _, _, _ = tracked_objects[j]
            dist = math.hypot((x2 - x1), (y2 - y1))
            print(f"Distance between object {i} and object {j}: {dist}")

    # cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    # cv2.putText(frame, str('1'), (504, 471), cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 0, 0), 1)

    # cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    # cv2.putText(frame, str('2'), (466, 485), cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 0, 0), 1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
