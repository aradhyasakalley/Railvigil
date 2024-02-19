import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import supervision as sv
import os
# Function to process video and count zones
def process_video(video_file):
    model = YOLO("yolov8l.pt")

    # Temporary file to store the uploaded video
    temp_file_path = "temp_video.mp4"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(video_file.read())

    cap = cv2.VideoCapture(temp_file_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    # Calculate zone width and height for four equal parts
    zone_width = frame_width // 2
    zone_height = frame_height // 2

    ZONE_POLYGONS = [
        np.array([[0, 0], [zone_width, 0], [zone_width, zone_height], [0, zone_height]]),
        np.array([[0, zone_height], [zone_width, zone_height], [zone_width, frame_height], [0, frame_height]]),
        np.array([[zone_width, 0], [frame_width, 0], [frame_width, zone_height], [zone_width, zone_height]]),
        np.array([[zone_width, zone_height], [frame_width, zone_height], [frame_width, frame_height], [zone_width, frame_height]]),
    ]

    zones = [
        sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=(zone_width, zone_height))
        for zone_polygon in ZONE_POLYGONS
    ]

    zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone,
            color=sv.Color.red(),
            thickness=2,
            text_thickness=4,
            text_scale=2
        )
        for zone in zones
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)
        labels = []
        for detection in detections:
            if len(detection) == 4:
                _, confidence, class_id, _ = detection
                label = f"{model.model.names[class_id]} {confidence:0.2f}"
                labels.append(label)
            else:
                print(f"Issue with detection: {detection}")

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        for i, zone_annotator in enumerate(zone_annotators):
            zone_annotator.annotate(scene=frame)
            zones[i].trigger(detections=detections)

        yield frame

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    os.remove(temp_file_path)

    model = YOLO("yolov8l.pt")

    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    # Calculate zone width and height for four equal parts
    zone_width = frame_width // 2
    zone_height = frame_height // 2

    ZONE_POLYGONS = [
        np.array([[0, 0], [zone_width, 0], [zone_width, zone_height], [0, zone_height]]),
        np.array([[0, zone_height], [zone_width, zone_height], [zone_width, frame_height], [0, frame_height]]),
        np.array([[zone_width, 0], [frame_width, 0], [frame_width, zone_height], [zone_width, zone_height]]),
        np.array([[zone_width, zone_height], [frame_width, zone_height], [frame_width, frame_height], [zone_width, frame_height]]),
    ]

    zones = [
        sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=(zone_width, zone_height))
        for zone_polygon in ZONE_POLYGONS
    ]

    zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone,
            color=sv.Color.red(),
            thickness=2,
            text_thickness=4,
            text_scale=2
        )
        for zone in zones
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)
        labels = []
        for detection in detections:
            if len(detection) == 4:
                _, confidence, class_id, _ = detection
                label = f"{model.model.names[class_id]} {confidence:0.2f}"
                labels.append(label)
            else:
                print(f"Issue with detection: {detection}")

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        for i, zone_annotator in enumerate(zone_annotators):
            zone_annotator.annotate(scene=frame)
            zones[i].trigger(detections=detections)

        yield frame

# Streamlit UI
def main():
    st.title("YOLOv8 Live Zone Detection")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
    if uploaded_file is not None:
        st.video(uploaded_file)
        st.write("Zone Counts:")
        video_stream = process_video(uploaded_file)
        for frame in video_stream:
            st.image(frame, channels="BGR", use_column_width=True)

def main():
    st.title("YOLOv8 Live Zone Detection")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
    if uploaded_file is not None:
        st.video(uploaded_file)
        st.write("Zone Counts:")
        
        # Create a container for the carousel
        carousel_container = st.empty()

        video_stream = process_video(uploaded_file)
        
        # Iterate through the video stream and add frames to the carousel
        for frame in video_stream:
            # Add frame to carousel
            carousel_container.image(frame, channels="BGR", use_column_width=True)

if __name__ == "__main__":
    main()
