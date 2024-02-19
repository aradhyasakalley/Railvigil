import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import math
import tempfile
import os

# COCO class labels
coco_classes = """
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
"""

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
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

def main():
    st.title('Coach Occupancy estimation')

    video_file = st.file_uploader('Upload a video file', type=['mp4'])  # File uploader widget

    if video_file is not None:
        # Save the uploaded video file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(video_file.read())

        # Get the path to the temporary file
        video_path = temp_file.name

        model = YOLO('yolov8s.pt')
        area1 = [(360,499),(470,498),(840,171),(835,169)]
        area2 = [(470,498),(550,499),(846,176),(840,171)]
        cap = cv2.VideoCapture(video_path)
        class_list = coco_classes.strip().split("\n")
        tracker = Tracker()
        count = 0
        while True:    
            ret,frame = cap.read()
            if not ret:
                break
            count += 1
            if count % 2 != 0:
                continue
            frame = cv2.resize(frame,(1020,500))
            results = model.predict(frame)
            a = results[0].boxes.data
            px = pd.DataFrame(a).astype("float")
            list = []
            for index,row in px.iterrows():
                x1=int(row[0])
                y1=int(row[1])
                x2=int(row[2])
                y2=int(row[3])
                d=int(row[5])
                c=class_list[d]
                if 'person' in c:
                   list.append([x1,y1,x2,y2])
            bbox_id = tracker.update(list)
            for bbox in bbox_id:
                   x3,y3,x4,y4,id = bbox
                   results = cv2.pointPolygonTest(np.array(area2,np.int32),((x4,y4)),False)
                   if results>=0:
                       cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
                   if id in tracker.center_points:
                       results1 = cv2.pointPolygonTest(np.array(area1,np.int32),((x4,y4)),False)
                       if results1>=0:
                            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                            cv2.circle(frame,(x4,y4),5,(255,0,255),-1)
                            cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
            cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
            cv2.putText(frame,str('1'),(504,471),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)
            cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)
            cv2.putText(frame,str('2'),(466,485),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)
            st.image(frame, channels="BGR")

if __name__ == '__main__':
    main()
