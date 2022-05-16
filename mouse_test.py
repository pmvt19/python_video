import cv2 
import numpy as np 
import time 

mouse_position = [None, None]

def mouse_move(event, x, y, flags, param):
    global mouse_position
    if event == cv2.EVENT_MOUSEMOVE:
        # print("x and y:", x, y)
        mouse_position = [x, y]
        # cv2.circle(img, (x, y), 5, 5)

    # if event == cv2.EVENT_LBUTTONDOWN:
    #     print("event: EVENT_LBUTTONDOWN")

size_x = 600
size_y = 600
color = (255, 0, 0)

prev_frame_time = 0
new_frame_time = 0


img = np.zeros((size_x, size_y, 3))
cv2.imshow('image', img)
cv2.setMouseCallback('image', mouse_move)
fps_list = []
while True:

    frame = img.copy() 
    x, y = mouse_position

    if (x != None and y != None):

        x = int(x)
        y = int(y)

        strXY = "{}, {}".format(x, y)

        cv2.putText(frame, strXY, (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        cv2.line(frame, (0, y), (size_x, y), (0, 0, 255), 1)
        cv2.line(frame, (x, 0), (x, size_y), (0, 255, 0), 1)
        cv2.circle(frame, (x, y), 12, color, -1)

    new_frame_time = time.time()
    time_diff = new_frame_time-prev_frame_time

    if time_diff > 0:
        fps = 1/(new_frame_time-prev_frame_time)
    else:
        fps = 0

    prev_frame_time = new_frame_time

    fps_list.append(fps)

    frames_per_second = "FPS: " + str(fps)

    cv2.putText(frame, frames_per_second, (500, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow('image', frame)

    if cv2.waitKey(1) == ord('q'):
        break 
    # time.sleep(0.5)
# cv2.imshow('image', img)
# cv2.setMouseCallback('image', mouse_move)

fps_list = np.array(fps_list)
avg_fps = np.average(fps_list)
max_fps = np.max(fps_list)
min_fps = np.min(fps_list)
print("Average FPS: {}".format(avg_fps))
print("Maximum FPS: {}".format(max_fps))
print("Minimum FPS: {}".format(min_fps))

cv2.destroyAllWindows()