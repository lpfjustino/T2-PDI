# Nome: Luis Paulo Falchi Justino
# NUSP: 8937479

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from enum import Enum

class Filter(Enum):
    L = 1
    G = 2
    H = 3
    S = 4

def enhance(filename, gamma, a, b, imshow):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Convert the image to 32bit float
    img_f = np.float32(img)

    # List of enhanced images
    # Keep the original image for comparison sakes
    enhanced_images = [img]

    # Applying different enhancement methods
    log_enhanced = imLog(img_f)
    enhanced_images.append(log_enhanced)

    gamma_enhanced = imGamma(img_f, gamma)
    enhanced_images.append(gamma_enhanced)

    hist_enhanced = imEqualHist(img_f)
    enhanced_images.append(hist_enhanced)

    sharp_enhanced = imSharp(img_f, (a,b))
    enhanced_images.append(sharp_enhanced)

    print("RMDS")
    for i, e_img in enumerate(enhanced_images):
        # Skip the source image
        if i == 0:
            continue
        print(Filter(i).name,"=",RMDS(img, e_img))

    # Shows images and their respective histograms if requested
    if imshow:
        show_images(enhanced_images)
        show_histograms(enhanced_images)


# Applies the Logarithmic filter to a given image f
def imLog(g):
    _, R, _, _ = cv2.minMaxLoc(g)
    c = 255/np.log(1+R)

    enh_img = cv2.log(g, 1 + g)
    enh_img = cv2.multiply(enh_img, c)
    enh_img = cv2.convertScaleAbs(enh_img)
    enh_img = cv2.normalize(enh_img, enh_img, 0, 255, cv2.NORM_MINMAX)

    return enh_img


# Applies the Gamma filter to a given image f
def imGamma(f, y):
    enh_img = cv2.pow(f, y)
    enh_img = cv2.convertScaleAbs(enh_img)
    enh_img = cv2.normalize(enh_img, enh_img, 0, 255, cv2.NORM_MINMAX)
    return enh_img


# Applies the Histogram Equalization filter to a given image f
def imEqualHist(f):
    # Extracting the global maximum, width and height
    _, L, _, _ = cv2.minMaxLoc(f)
    M = np.array(f).shape[0]
    N = np.array(f).shape[1]

    # Transform the matrix in a row vector
    f_array = np.array(f).flatten()

    hist = imHistogram(f_array)
    hist = myCumHist(hist)

    for i in range(len(f_array)):
        val = int(f_array[i]) % 254
        f_array[i] = ((L-1)/(M*N))*hist[val]

    enh_img = f_array.reshape(M, N)
    enh_img = cv2.convertScaleAbs(enh_img)
    enh_img = cv2.normalize(enh_img, enh_img, 0, 255, cv2.NORM_MINMAX)

    return enh_img


# Applies the Sharpening filter to a given image f
def imSharp(f,y):
    # Filter
    w = np.array([0.05, 0.1, 0.05, 0.1, 0.4, 0.1, 0.05, 0.1, 0.05])

    # Image with borders in order to make it easier to apply the filter
    b_f = cv2.copyMakeBorder(f, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    # Coordinates of the outer edges
    bottom_edge = b_f.shape[0]
    right_edge = b_f.shape[1]

    # Applying the filter
    b_xy = np.ndarray(f.shape)
    for (i, j), v in np.ndenumerate(b_f):
        # Skip borders
        if i==0 or i==bottom_edge-1 or j==0 or j==right_edge-1:
            continue

        z = b_f[i-1:i+2, j-1:j+2].flatten()
        b_xy[i-1,j-1] = np.dot(z, w)

    a = y[0]
    b = y[1]

    enh_img = cv2.multiply(f, a) + cv2.multiply(cv2.subtract(b,f), b)
    enh_img = cv2.convertScaleAbs(enh_img)
    enh_img = cv2.normalize(enh_img, enh_img, 0, 255, cv2.NORM_MINMAX)

    return enh_img


# Personal implementation of the histogram algorithm given an array
def imHistogram(array):
    hist = np.zeros(255, dtype=int)

    array_f = array.flatten()

    for element in array_f:
        idx = math.floor(element) % 255
        hist[idx] += 1

    return hist


# Personal implementation of the cumulative sum algorithm given a histogram
def myCumHist(h):
    ch = []
    ch.append(h[0])

    for i in range(len(h)-1):
        i += 1
        ch.append(ch[i-1] + h[i])

    return ch

    # np cumsum
    #return np.cumsum(array)


# Presents a given histogram on the screen
def showHistogram(h):
    plt.plot(range(255), h, "r--", linewidth=1)
    plt.show()


# Shows all given images in a list
def show_images(images):
    # Showing all enhanced images
    for image in images:
        cv2.imshow("", image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


# Shows all given images' histogram in a list
def show_histograms(imgs):
    # Showing all enhanced images' histogram
    for img in imgs:
        showHistogram(imHistogram(img))


def RMDS(f,g):
    # Ignore the error that says overflow error mays happen
    np.seterr(all='ignore')

    eps = 0

    array_f = f.flatten()
    array_g = g.flatten()

    MN = array_f.shape[0]

    for i, _ in enumerate(array_f):
        eps += (array_f[i] - array_g[i])


    eps /= float(MN)
    eps = np.sqrt(eps)

    return eps

# enhance('arara.jpg', 0.8, 0.7, 0.3, 1)
# enhance('nap.jpg', 0.8, 0.7, 0.3, 0)
# enhance('cameraman.png', 0.8, 0.7, 0.3, 0)

enhance(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]),int(sys.argv[5]))