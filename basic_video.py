import cv2 
import numpy as np 

def bgr_to_hsv(color):
    color = np.array([[color]])
    # print(color.shape)
    color = np.float32(color)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return hsv[0, 0]

xx = np.arange(360)
yy = np.arange(640)

nsx, nsy = np.meshgrid(xx, yy)
    
# Start the Video Capture (Argument is which camera to use)
cap = cv2.VideoCapture(0)

while True:

    # Read a frame 
    ret, frame = cap.read() 

    # Extract the Dimensions of the frame
    width = int(cap.get(3))
    height = int(cap.get(4))

    frame = cv2.resize(frame, (640, 360))
    frame = np.flip(frame, axis=1)

    # Convert the image from BGR pixels to HSV pixels
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_brown = bgr_to_hsv([0, 33, 48])
    upper_brown = bgr_to_hsv([18, 105, 145])

    # lower_red = bgr_to_hsv([0, 0, 255])
    # upper_red = bgr_to_hsv([12, 12, 99])

    # lower_red = bgr_to_hsv([12, 12, 99])
    # upper_red = bgr_to_hsv([0, 0, 255])

    lower_red = np.array([160,20,70])
    upper_red = np.array([190,255,255])




    # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # mask = cv2.inRange(hsv, lower_brown, upper_brown)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # red_mask = cv2.inRange(hsv, lower_red, upper_red)
    # mask = cv2.bitwise_or(blue_mask, red_mask)
    # result = cv2.bitwise_and(frame, frame, mask=mask)
    result = cv2.bitwise_and(frame, frame, mask=blue_mask)

    x_in_y = np.not_equal(np.array([0, 0, 0]).reshape(1, 1, 3), result).all(axis=2)
    idx = np.array(np.where(x_in_y))

    # x = np.array(x)
    # y = np.array(y)

    x = idx[0]
    y = idx[1]

    if (x.shape == y.shape) and (x.shape[0] > 0 and y.shape[0] > 0):
        x_avg = int(np.average(x))
        y_avg = int(np.average(y))

        result = cv2.circle(result.copy(), (int(y_avg), int(x_avg)), 5, (255, 255, 255), 15)

    cv2.imshow('newFrame', frame)
    cv2.imshow('frame', result) 


    if cv2.waitKey(1) == ord('q'):
        break 

cap.release() 
cv2.destroyAllWindows() 