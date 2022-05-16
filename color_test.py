import cv2 
import numpy as np 

def bgr_to_hsv(color):
    color = np.array([[color]])
    # print(color.shape)
    color = np.float32(color)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return hsv[0, 0]

# xx = np.arange(360)
# yy = np.arange(640)

# nsx, nsy = np.meshgrid(xx, yy)

# img = np.zeros((360, 640, 3))


# lower_blue = np.array([110, 50, 50])
# upper_blue = np.array([130, 255, 255])

# img[nsx, nsy] = upper_blue 
# img = img.astype(np.uint8)

# img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

xx = np.arange(100)
yy = np.arange(100)

nsx, nsy = np.meshgrid(xx, yy)

img = np.zeros((100, 100, 3))


lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

img[nsx, nsy] = upper_blue 

img = img.astype(np.uint8)
hsv_color = bgr_to_hsv([255, 0, 85])

img[nsx, nsy] = hsv_color 

img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)





# print("TO HSV:", bgr_to_hsv([255, 0, 80]))

# print("IMG: ", img)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


