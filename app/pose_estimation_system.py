import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 pose estimation model
model = YOLO('yolov8s-pose.pt')  # Ensure you have downloaded the model

# Input and output paths
input_video_path = './/..//res//videos//example.mp4'  # Replace with your video path
output_video_path = './/..//res//videos//example_result.mp4'  # Output path

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print("Processing video...")
frame_count=0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1  
    # Run pose estimation
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
     # Add frame counter to the top-right corner
    '''
    text = f"Frame: {frame_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (139, 0, 0)  # Dark blue in BGR
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = frame_width - text_size[0] - 10
    text_y = 30
    cv2.putText(annotated_frame, text, (text_x, text_y), font, font_scale, color, thickness)
    '''
    movements = 0
    if frame_count>145:
        text = f"Movement: {movements}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (139, 0, 0)  # Dark blue in BGR
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = frame_width - text_size[0] - 10
        text_y = 30
        cv2.putText(annotated_frame, text, (text_x, text_y), font, font_scale, color, thickness)

    # Write the frame to the output video
    out.write(annotated_frame)

    # Display the frame (optional)
    cv2.imshow('Pose Estimation', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved at:", output_video_path)
