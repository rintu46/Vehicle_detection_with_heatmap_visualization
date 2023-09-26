from ultralytics import YOLO
import cv2
import math
import numpy as np


# for now it's work, need to work more for future task
 

model = YOLO('yolov8m.pt')
print('ok',model)




video_file = '/home/parvej/Sikder Md Saiful Islam/september/nw 3dnet/nw3/heatmap visualization/data/roads_-_10812 (720p).mp4'  # Replace with your video file's path

# Open the video file
cap = cv2.VideoCapture(video_file)

# Set the size of the resized frame
resize_width = 1200
resize_height = 700

global_img_array = None
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# print(w)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

global_img_array = np.ones([int(h), int(w)], dtype = np.uint32)
# print('global_img_array', global_img_array)
# print('global_img_array', global_img_array.shape)



while True:
    success, frame = cap.read()
    
    rs = model.predict(frame,classes=2, conf=0.02)
    # print(rs)

    for r in rs:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            print('hereeeeeee',x1,y1,x2,y2)
            print('hereeeeeee',frame)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255),3)
            global_img_array[y1:y2, x1:x2] +=1
            print("gia", global_img_array)
            print("gia", global_img_array.shape)
        global_img_array_norm = (global_img_array - global_img_array.min()) / (global_img_array.max() - global_img_array.min()) * 255
        global_img_array_norm = global_img_array_norm.astype('uint8')
        global_img_array_norm = cv2.GaussianBlur(global_img_array_norm, (9,9), 0)
        heatmap_img = cv2.applyColorMap(global_img_array_norm, cv2.COLORMAP_JET)
        super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, frame, 0.5, 15)


            # conf = math.ceil((box.conf[0]*10))/100
            # cls = int(box.cls[0])
            # class_name = className
        # print(boxes)




     # Resize the frame before displaying it
    frame = cv2.resize(super_imposed_img, (resize_width, resize_height))
    # Display the frame
    cv2.imshow('root', frame)

    # Check for the 'q' key to quit the video
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break



# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()