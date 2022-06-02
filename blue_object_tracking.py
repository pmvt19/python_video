from sklearn.cluster import KMeans 
from sklearn.preprocessing import MinMaxScaler
import cv2 
import numpy as np 
import time 
import matplotlib.pyplot as plt
import datetime 
import math 

def bgr_to_hsv(color):
    color = np.array([[color]])
    # print(color.shape)
    color = np.float32(color)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return hsv[0, 0]

def calc_dist(x1, y1, a, b, c):
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
    return d 

xx = np.arange(360)
yy = np.arange(640)

nsx, nsy = np.meshgrid(xx, yy)
    
# Start the Video Capture (Argument is which camera to use)
cap = cv2.VideoCapture(0)
time.sleep(1)

j = 0
while j == 0:
    j += 1
    ret, frame = cap.read() 
    frame = cv2.resize(frame, (640, 360))
    frame = np.flip(frame, axis=1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255]) 
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(frame, frame, mask=blue_mask)

    kernel = np.ones((1, 1), np.uint8)
    result = cv2.erode(result, kernel, iterations=1)

    x_in_y = np.not_equal(np.array([0, 0, 0]).reshape(1, 1, 3), result).all(axis=2)
    idx = np.array(np.where(x_in_y))

    # print("idx shape:", idx.shape)

    x = idx[0]
    y = idx[1]

    dist = [] 
    K = range(1, 5)
    for clust in K: 
        k_model = KMeans(n_clusters=clust)
        k_model.fit(idx.T)
        dist.append(k_model.inertia_)
    # plt.plot(K, dist)
    # plt.show() 

    a = dist[0] - dist[3]
    b = K[3] - K[0] 
    c1 = K[0] * dist[3]
    c2 = K[3] * dist[0] 
    c = c1 - c2 

    lineDist = [] 

    for k in range(4):
        lineDist.append(calc_dist(K[k], dist[k], a, b, c))
    
    # plt.plot(K, lineDist)
    # plt.show() 

    km = KMeans(n_clusters=2)

    print("Recommended Number of Trackers: ", np.argmax(lineDist))

    scaler = MinMaxScaler() 

    # idx = scaler.fit_transform(idx.T).T
    

    start = datetime.datetime.now()
    y_predicted = km.fit_predict(idx.T)
    end = datetime.datetime.now()

    # idx = scaler.inverse_transform(idx.T).T

    # print(y_predicted)

    # print("Type:", type(y_predicted))

    clusters = np.unique(y_predicted) 

    # plt.scatter(idx[0], idx[1])
    # plt.show()


    # print("Clusters:", clusters)
    # print("Ind: ", idx, idx.shape)
    colors = ['green', 'red', 'black', 'blue']

    for i in range(len(clusters)):
        cluster = clusters[i] 
        # print("Y_Pred shape:", y_predicted.shape)
        # new_ind = np.where(y_predicted, cluster)

        

        new_ind = np.argwhere(y_predicted == cluster)
        print("Len: ", new_ind.shape)
        # if new_ind.shape[0] < 1200:
        #     continue 
        
        myX = idx[0, new_ind]
        myY = idx[1, new_ind]
        plt.scatter(myX, myY, color=colors[i])

        if (myX.shape == myY.shape) and (myX.shape[0] > 0 and myY.shape[0] > 0):
            x_avg = int(np.average(myX))
            y_avg = int(np.average(myY))

            result = cv2.circle(result.copy(), (int(y_avg), int(x_avg)), 5, (255, 255, 255), 15)
    plt.show()


    cv2.imshow('frame', frame)
    cv2.imshow('newframe', result)


    if cv2.waitKey(1) == ord('q'):
        break 



# print("Time: ", end - start)

# cv2.waitKey(0)

cap.release() 
cv2.destroyAllWindows() 