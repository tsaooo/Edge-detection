import cv2
import numpy as np
from pathlib import Path
import func

path1 = "/Users/eehenry/Documents/tsao/DIP_project2/images/p1im1.png"
path2 = "/Users/eehenry/Documents/tsao/DIP_project2/images/p1im4.png"
path3 = "/Users/eehenry/Documents/tsao/DIP_project2/images/p1im5.png"
path4 = "/Users/eehenry/Documents/tsao/DIP_project2/images/p1im6.png"
w_path = ['canny_img1.png','canny_img4.png','canny_img5.png','soble50_img6.png']
paths = [path1, path2, path3, path4]
size = 5
T = 60
'''for i in range(4):
    img = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
    if i == 3 : size = 13
    img = func.median_filter(img,size=size)
    new_img = func.canny(img)
    cv2.imwrite(w_path[i], new_img)'''
img = cv2.imread(paths[3], cv2.IMREAD_GRAYSCALE)
img = func.median_filter(img, 15)
#img = func.sobel(img)
#new_img = func.threshold(new_img, T = 40)
new_img = func.canny(img)
cv2.imwrite(w_path[3], new_img)