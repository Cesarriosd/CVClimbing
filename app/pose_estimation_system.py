import cv2
import torch
from ultralytics import YOLO
from math import dist
import numpy as np

# Load YOLOv8 pose estimation model
model = YOLO('yolov8l-pose.pt')  # Ensure you have downloaded the model

# Input and output paths
input_video_path = './/..//res//videos//source//example.mp4'  # Replace with your video path
output_video_path = './/..//res//videos//results//example_result.mp4'  # Output path

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
movements = 0
whist = False
left_whist = (0,0)
right_whist_last = (0,0)
left_whist_last = (0,0)
right_whist_last = (0,0)
start_frame =  146
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1  
    # Run pose estimation
    results = model(frame, conf=0.7)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
    #=========================================================
    ##Need to define the thresholds
    #=========================================================
    #left_whist_conf_tresh = [keypoints_conf[i] for i in [9, 10, 15, 16]]
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
    #left_whist_last, right_whist_last, left_ankle_last, right_ankle_last=torch.ones(4)
    if frame_count >= start_frame and whist == False:
        ## Initial Keypoints
        try:
            keypoints_conf = results[0].keypoints.conf[0]
            keypoints_yolo = results[0].keypoints.xy[0]
            left_whist_last = keypoints_yolo[9] if keypoints_yolo[9][0]>0 and keypoints_yolo[9][1]>0 and keypoints_conf[9]>0.7  else left_whist_last
            right_whist_last = keypoints_yolo[10] if keypoints_yolo[10][0]>0 and keypoints_yolo[10][1]>0 and keypoints_conf[10]>0.7 else right_whist_last
            #, right_whist_last, left_ankle_last, right_ankle_last =[keypoints_yolo[i] for i in [9, 10, 15, 16]]
        except: 
            None
        if all(x > 0 for x in left_whist_last) and all(x > 0 for x in right_whist_last):
            whist = True
    elif frame_count > start_frame and whist == True:
        #define the var to store the last position of the keypoints of interest
    
        #c_w= results[0].keypoints.conf[0][9]> 0.7
        #print(1 if c_w.numpy()==True else 0)
        #extract c
        try:
            keypoints_conf = results[0].keypoints.conf[0]
            keypoints_yolo = results[0].keypoints.xy[0]
            left_whist = keypoints_yolo[9] if keypoints_yolo[9][0]>0 and keypoints_yolo[9][1]>0 and keypoints_conf[9]>0.7  else left_whist
            right_whist = keypoints_yolo[10] if keypoints_yolo[10][0]>0 and keypoints_yolo[10][1]>0 and keypoints_conf[10]>0.7 else right_whist
            #left_whist_conf, right_whist_conf, left_ankle_conf, right_ankle_conf = [keypoints_conf[i] for i in [9, 10, 15, 16]]
            #left_whist, right_whist, left_ankle, right_ankle = [keypoints_yolo[i] for i in [9, 10, 15, 16]]
        except:
            None
        if all(x > 0 for x in left_whist) and all(x > 0 for x in right_whist):
            distance_whist_last =  dist(left_whist_last,right_whist_last ) if dist(left_whist_last,right_whist_last )>0 else 1
            distance_whist =  dist(left_whist, right_whist)
            distance_whist_change =  np.round(np.abs((distance_whist - distance_whist_last)/distance_whist_last),2)
            if distance_whist_change>0.8:
                movements+=1
            if keypoints_yolo[9][0]>0 and keypoints_yolo[9][1]>0:
                left_whist_last =keypoints_yolo[9]
            if keypoints_yolo[10][0]>0 and keypoints_yolo[10][1]:
                right_whist_last =keypoints_yolo[10]
        
        #left_whist_last, right_whist_last, left_ankle_last, right_ankle_last =[keypoints_yolo[i] for i in [9, 10, 15, 16]]

        ##Distance annotations
        text_whist = f"Whist movement: "#{distance_whist_change}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 250, 250)  # Dark blue in BGR
        thickness = 2
        text_size = cv2.getTextSize(text_whist, font, font_scale, thickness)[0]
        text_x = frame_width - text_size[0] - 10
        text_y = frame_height-50
        cv2.putText(annotated_frame, text_whist, (text_x, text_y), font, font_scale, color, thickness)
        ##########
        text_whist = f"{distance_whist_change}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 250, 250)  # Dark blue in BGR
        thickness = 2
        text_size = cv2.getTextSize(text_whist, font, font_scale, thickness)[0]
        text_x = frame_width - text_size[0] - 15
        text_y = frame_height-20
        cv2.putText(annotated_frame, text_whist, (text_x, text_y), font, font_scale, color, thickness)
        ## Movements annotations
        """
        text = f"Movement: {movements}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (139, 0, 0)  # Dark blue in BGR
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = frame_width - text_size[0] - 10
        text_y = 30
        cv2.putText(annotated_frame, text, (text_x, text_y), font, font_scale, color, thickness)
        """

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
