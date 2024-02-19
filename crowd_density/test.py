import cv2

# Read the video file
video_path = 'classroom.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video's frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the zones
zone1 = (0, 0, frame_width // 2, frame_height // 2)
zone2 = (frame_width // 2, 0, frame_width, frame_height // 2)
zone3 = (0, frame_height // 2, frame_width // 2, frame_height)
zone4 = (frame_width // 2, frame_height // 2, frame_width, frame_height)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = 'output_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Divide the frame into zones
    zone1_frame = frame[zone1[1]:zone1[3], zone1[0]:zone1[2]]
    zone2_frame = frame[zone2[1]:zone2[3], zone2[0]:zone2[2]]
    zone3_frame = frame[zone3[1]:zone3[3], zone3[0]:zone3[2]]
    zone4_frame = frame[zone4[1]:zone4[3], zone4[0]:zone4[2]]

    # Draw rectangles to outline the zones
    cv2.rectangle(frame, (zone1[0], zone1[1]), (zone1[2], zone1[3]), (255, 0, 0), 2)
    cv2.rectangle(frame, (zone2[0], zone2[1]), (zone2[2], zone2[3]), (0, 255, 0), 2)
    cv2.rectangle(frame, (zone3[0], zone3[1]), (zone3[2], zone3[3]), (0, 0, 255), 2)
    cv2.rectangle(frame, (zone4[0], zone4[1]), (zone4[2], zone4[3]), (255, 255, 0), 2)

    # Combine frames
    combined_frame = cv2.vconcat([cv2.hconcat([zone1_frame, zone2_frame]), cv2.hconcat([zone3_frame, zone4_frame])])

    # Display the frame
    cv2.imshow('Frame', combined_frame)

    # Write the frame to the output video
    out.write(combined_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
