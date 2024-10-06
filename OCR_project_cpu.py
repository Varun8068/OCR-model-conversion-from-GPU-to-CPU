import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
import paddle
import time  


print(f"Using GPU: {paddle.is_compiled_with_cuda()}")  # This will still show if the system supports GPU, but we won't use it

# Initialize PaddleOCR with CPU (use_gpu=False)
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# Video input/output paths
input_video_path = "/content/OCR-Sample.mp4" 
output_video_path = "output_video_cpu.mp4"  

# Capture video from file
cap = cv2.VideoCapture(input_video_path)

# Get video properties (FPS, width, height, frame count)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input video FPS: {fps}")
print(f"Video resolution: {width}x{height}")
print(f"Total frames: {frame_count}")

# Define codec and create VideoWriter object for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Variables for tracking frame number, accuracies, and FPS
frame_num = 0
accuracies = []
fps_values = []

start_time = time.time()

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR frame to RGB for PaddleOCR
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run OCR on the current frame (CPU version)
    result = ocr.ocr(img_rgb)
    total_confidence = 0
    num_words = 0  

    if result is not None:
        for line in result:
            if line:  
                for word_info in line:
                    text = word_info[-1][0]  # Extract detected text
                    confidence = word_info[-1][1]  # Extract confidence score
                    total_confidence += confidence
                    num_words += 1

                    bbox = word_info[0]

                    # Convert bbox points to integer and draw them on the frame
                    bbox = [(int(point[0]), int(point[1])) for point in bbox]
                    cv2.polylines(frame, [np.array(bbox)], True, (0, 255, 0), 2)

                    # Put the detected text on the frame
                    cv2.putText(frame, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Write processed frame to output video
    out.write(frame)

    # Calculate average accuracy for this frame
    if num_words > 0:
        accuracy = total_confidence / num_words  
    else:
        accuracy = 0

    accuracies.append(accuracy)

    # Track frame number and calculate FPS
    frame_num += 1
    elapsed_time = time.time() - start_time
    current_fps = frame_num / elapsed_time
    fps_values.append(current_fps)   

    # Print progress
    print(f"Processing frame {frame_num}/{frame_count}, Accuracy: {accuracy:.2f}, FPS: {current_fps:.2f}")

end_time = time.time()

# Calculate total time and average FPS
total_time = end_time - start_time
average_fps = frame_num / total_time

print(f"Total time taken for processing: {total_time:.2f} seconds")
print(f"Average FPS during processing: {average_fps:.2f}")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed and saved to", output_video_path)