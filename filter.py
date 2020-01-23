
import os 
import cv2 
import numpy as np

def detect_pictures(directory):
    pics = [directory+item for item in os.listdir(directory)]
    for pic in pics:
            filter_image(pic)


def filter_image(path, clusers = 3):
    k = clusers
    kernel = np.ones((9,9),np.uint8)
    img = cv2.imread(path)
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    closing = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel,iterations=1)
    opening = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel,iterations=1)
    _, thresh = cv2.threshold(opening, 170, 255, cv2.THRESH_BINARY)
    contours, _  = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key= cv2.contourArea, reverse=True)[:2]
    cv2.fillPoly(gray, pts =contours, color=(255,255,255))
    cv2.imshow(" ", gray)
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
    res = cv2.bitwise_and(img,img,mask = thresh)
    
    #should be saved 
    cv2.imshow('final',res)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

detect_pictures('./visble/thighes/')