import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2 
import numpy as np 
import time 
from pytorch_classes import CLASSES


# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
# model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
model.eval()

output = ['boxes', 'labels', 'scores']

cap = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0


while True:

    # Read a frame 
    ret, frame = cap.read() 
    frame = cv2.resize(frame, (640, 360))

    frame_copy = frame.copy()

    # Extract the Dimensions of the frame
    width = int(cap.get(3))
    height = int(cap.get(4))

    frame = cv2.resize(frame, (640, 360))
    frame = np.flip(frame, axis=1)
    

    frame = frame.transpose((2, 0, 1))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)

    # frame = frame.double()

    # print("Frame Size:", frame.shape)

    # frame = torch.from_numpy(frame.copy())
    # frame = frame.double()
    frame = torch.FloatTensor(frame)
    # print("Frame Size After Conversion:", type(frame))
    frame_copy = np.flip(frame_copy, axis=1)
    with torch.no_grad():
        detections = model(frame)

        detections = detections[0]

        # print("Sizes of outputs:", detections[output[0]].size(), detections[output[1]].size(), detections[output[2]].size())

        num_objects_detected = detections['labels'].size()[0]

        boxes = detections['boxes']
        labels = detections['labels']
        scores = detections['scores']

        for i in range(num_objects_detected):
            if scores[i] > 0.7:
                idx = int(detections["labels"][i])
                start_x, start_y, end_x, end_y = detections['boxes'][i].detach().numpy().astype(int)
                frame_copy = cv2.rectangle(frame_copy.copy(), (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
                cv2.putText(frame_copy, str(CLASSES[idx]), (start_x, start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # print("Cur Object:", labels[i])


        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        frames_per_second = "FPS: " + str(fps)

        cv2.putText(frame_copy, frames_per_second, (500, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


        # print("Len output:", len(detections))
        
        cv2.imshow('newFrame', frame_copy)

    if cv2.waitKey(1) == ord('q'):
        break 

cap.release() 
cv2.destroyAllWindows() 
