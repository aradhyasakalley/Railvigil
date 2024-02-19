import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import math
import tempfile
import os

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for obj_id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[obj_id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, obj_id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDs not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, obj_id = obj_bb_id
            center = self.center_points[obj_id]
            new_center_points[obj_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

# Function to perform object detection and tracking
def perform_detection_and_tracking(video_file):
    # Save the uploaded video file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(video_file.read())

    # Get the path to the temporary file
    video_path = temp_file.name

    # Initialize YOLO model
    model = YOLO('yolov8s.pt')

    # Define areas of interest
    area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]
    area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]

    cap = cv2.VideoCapture(video_path)

    # COCO class labels
    coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                    'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    classes_to_track = ['person', 'handbag', 'suitcase']  # Define the classes to track

    tracker = Tracker()  # Initialize the object tracker

    threshold_distance = 200  # Set a threshold distance to determine abandonment (adjust as needed)

    # Dictionary to store the initial positions of tracked bags/suitcases
    initial_positions = {}
    abandoned_objects_count = 0
    counted_abandoned_objects = {}

    st.header("Uploaded Video")
    st.video(video_file)

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
            c = coco_classes[d]
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

                    # Store initial positions of bags/suitcases
                    if class_label in ['handbag', 'suitcase'] and obj_id not in initial_positions:
                        initial_positions[obj_id] = (x1, y1, x2, y2)

        # Check for abandoned objects
        for obj_id, initial_pos in initial_positions.items():
            if obj_id < len(tracked_objects) and len(tracked_objects[obj_id]) >= 5:  # Check if obj_id is valid
                current_pos = tracked_objects[obj_id][:4]
                dist = math.hypot((current_pos[0] - initial_pos[0]), (current_pos[1] - initial_pos[1]))
                if obj_id not in counted_abandoned_objects and dist > threshold_distance:
                    abandoned_objects_count += 1  # Increment the count
                    counted_abandoned_objects[obj_id] = True  # Mark the object as counted
                    print(f"Abandoned object detected: {tracked_objects[obj_id]}")
        
        cv2.putText(frame, f'Abandoned Objects: {abandoned_objects_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)      

        st.header("Processed Video")
        st.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

    print(f"Total abandoned objects detected: {abandoned_objects_count}")

# Streamlit app code
def main():
    st.title('Abandoned Object Detection')

    video_file = st.file_uploader('Upload a video file', type=['mp4'])  # File uploader widget

    if video_file is not None:
        # Perform object detection and tracking on the uploaded video
        perform_detection_and_tracking(video_file)

if __name__ == '__main__':
    main()
