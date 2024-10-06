
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
import paddle
import time  


print(f"Using GPU: {paddle.is_compiled_with_cuda()}")

ocr = PaddleOCR(use_angle_cls=True, lang='en',use_gpu = True)

 

input_video_path = "/content/OCR-Sample.mp4" 
output_video_path = "output_video_gpu.mp4"  


cap = cv2.VideoCapture(input_video_path)


fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input video FPS: {fps}")
print(f"Video resolution: {width}x{height}")
print(f"Total frames: {frame_count}")


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_num = 0
accuracies = []
fps_values = []

start_time = time.time()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    result = ocr.ocr(img_rgb)
    total_confidence = 0
    num_words = 0  

    if result is not None:
        for line in result:
            if line:  
                for word_info in line:
                    text = word_info[-1][0]
                    confidence = word_info[-1][1]  
                    total_confidence += confidence
                    num_words += 1

                    bbox = word_info[0]

                    bbox = [(int(point[0]), int(point[1])) for point in bbox]
                    cv2.polylines(frame, [np.array(bbox)], True, (0, 255, 0), 2)


                    cv2.putText(frame, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    out.write(frame)

    if num_words > 0:
        accuracy = total_confidence / num_words  
    else:
        accuracy = 0

    accuracies.append(accuracy)


    frame_num += 1
    elapsed_time = time.time() - start_time
    current_fps = frame_num / elapsed_time
    fps_values.append(current_fps)   

    print(f"Processing frame {frame_num}/{frame_count}, Accuracy: {accuracy:.2f}, FPS: {current_fps:.2f}")

end_time = time.time()


total_time = end_time - start_time
average_fps = frame_num / total_time

print(f"Total time taken for processing: {total_time:.2f} seconds")
print(f"Average FPS during processing: {average_fps:.2f}")


cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed and saved to", output_video_path)
