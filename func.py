import numpy as np
import cv2

def gamma(img, g):
    '''img gamma correction'''
    new_img = ((img / 255.0) ** g * 255.0)
    return new_img
def histo_eql(img):
    '''img histogram equalization'''
    histogram = np.zeros(256)
    img1d = img.flatten()
    for p in img1d:
        histogram[p] += 1
    cdf = np.cumsum(histogram)
    cdf = np.around((cdf / cdf.max()) * 255)
    #cdf = cdf.astype('uint8')
    new_img = np.reshape(cdf[img1d], img.shape)
    return new_img
def median_filter(img, size = 3):
    '''img median blurring'''
    if img.ndim == 2 : gray = True
    else : gray = False
    pad = size // 2
    img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    new_img = np.zeros_like(img)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if gray:
                val = np.median(img_pad[x:x+size, y:y+size])
                new_img.itemset((x,y), val)
            else:
                for c in range(3):
                    val = np.median(img_pad[x:x+size, y:y+size, c])
                    new_img.itemset((x,y,c), val)
    return new_img

def laplacian_filter(img, size = 3):
    '''img laplacian sharpening'''
    kernel = np.array([[0,-1,0],
                       [-1,4,-1],
                       [0,-1,0]])

    if img.ndim == 2 : gray = True
    else : gray = False
    pad = size // 2
    img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    fil_img = np.zeros(img.shape)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if gray :
                val = np.sum(img_pad[x:x+size, y:y+size] * kernel)
                fil_img.itemset((x,y), val)
            else :
                for c in range(3):
                    val = np.sum(img_pad[x:x+size, y:y+size, c] * kernel)
                    fil_img.itemset((x,y,c), val)
    new_img = fil_img + img
    return new_img

def threshold(img, T = 60):
    for row in img :
        row[row < T] = 0
        row[row >= T] = 255
    return img

def prewitt(img, size = 3):
    '''prewitt gradient filter'''
    kernel_y = np.array([[-1,-1,-1],
                         [0,0,0],
                         [1,1,1]])
    kernel_x = np.array([[-1,0,1],
                         [-1,0,1],
                         [-1,0,1]])
    grad_x = conv(kernel_x, img, size)
    grad_y = conv(kernel_y, img, size)
    new_img = np.add(np.abs(grad_x), np.abs(grad_y))
    return new_img

def sobel(img, size = 3):
    '''sobel gradient filter'''
    kernel_y = np.array([[-1,-2,-1],
                         [0,0,0],
                         [1,2,1]])
    kernel_x = np.array([[-1,0,1],
                         [-2,0,2],
                         [-1,0,1]])
    grad_x = conv(kernel_x, img, size)
    grad_y = conv(kernel_y, img, size)
    new_img = np.add(np.abs(grad_x), np.abs(grad_y))
    return new_img

def log(img):
    kernel = np.array([[0,0,-1,0,0],
                       [0,-1,-2,-1,0],
                       [-1,-2,16,-2,-1],
                       [0,-1,-2,-1,0],
                       [0,0,-1,0,0]])
    filted = conv(kernel, img, 5)
    return filted                      

def conv(kernel, img, size):
    '''doing padding and conv operation to img base on given kernel and size'''
    if img.ndim == 2 : gray = True
    else : gray = False
    pad = size // 2
    img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    fil_img = np.zeros(img.shape)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if gray :
                val = np.sum(img_pad[x:x+size, y:y+size] * kernel)
                fil_img.itemset((x,y), val)
            else :
                for c in range(3):
                    val = np.sum(img_pad[x:x+size, y:y+size, c] * kernel)
                    fil_img.itemset((x,y,c), val)
    return fil_img

def canny(img):
    '''img: the gradient magnitude map'''
    kernel_y = np.array([[-1,-2,-1],
                         [0,0,0],
                         [1,2,1]])
    kernel_x = np.array([[-1,0,1],
                         [-2,0,2],
                         [-1,0,1]])
    grad_x = conv(kernel_x, img, size = 3)
    grad_y = conv(kernel_y, img, size = 3)
    mag = np.add(np.abs(grad_x), np.abs(grad_y))
    theta = np.arctan2(grad_y, grad_x) * 180 /np.pi
    mag = non_max_suppression(mag, theta)
    res = doub_threshold(mag)
    return res
def non_max_suppression(img, theta):
    res = np.zeros(img.shape)
    theta[theta < 0] += 180
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            try :
                if (0 <= theta[i,j] < 22.5) or (157.5 <= theta[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                elif (22.5 <= theta[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                elif (67.5 <= theta[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                elif (112.5 <= theta[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]
                if q > img[i,j] or r > img[i,j]: res[i,j] = 0
                else: res[i,j] = img[i,j]
            except IndexError:
                pass
    return res
def doub_threshold(img):
    T_h = 40
    T_l = 0
    new_img = np.zeros(img.shape)
    s_x, s_y = np.where(img >= T_h)
    w_x, w_y = np.where((img >= T_l) & (img < T_h))
    new_img[s_x, s_y] = 255
    change = True
    while(change):
        change = False
        for i in range(len(w_x)):
            for j in range(len(w_y)):
                x = w_x[i]
                y = w_y[j]
                if x != -1 and y != -1:
                    if x - 1 >= 0 and new_img[x-1,y] == 255:
                        change = True
                    if x + 1 < img.shape[0] and new_img[x+1,y] == 255: 
                        change = True
                    if y - 1 >= 0 and new_img[x,y-1] == 255: 
                        change = True
                    if y + 1 < img.shape[1] and new_img[x,y+1] == 255: 
                        change = True
                    if x - 1 >= 0 and y - 1 >=0 and new_img[x-1,y-1] == 255: 
                        change = True
                    if x + 1 < img.shape[0] and y - 1 >=0 and new_img[x+1,y-1] == 255: 
                        change = True
                    if x - 1 >= 0 and y + 1 < img.shape[1] and new_img[x-1,y+1] == 255: 
                        change = True
                    if x + 1 < img.shape[0] and y + 1 < img.shape[1] and new_img[x+1,y+1] == 255: 
                        change = True
                    if change:
                        new_img.itemset((x,y), 255)
                        w_x[i] = -1
                        w_y[j] = -1
    return new_img