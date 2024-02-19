import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import supervision as sv

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()


    cap = cv2.VideoCapture('classroom.mp4')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    

    model = YOLO("yolov8l.pt")

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

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()