# Nome: Luis Paulo Falchi Justino
# NUSP: 8937479

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys


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

    if imshow:
        show_images(enhanced_images)


def show_images(images):
    # Showing all enhanced images
    for image in images:
        cv2.imshow("", image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def imLog(g):
    _, R, _, _ = cv2.minMaxLoc(g)
    c = 255/np.log(1+R)

    enh_img = cv2.log(g, 1 + g)
    enh_img = cv2.multiply(enh_img, c)
    enh_img = cv2.convertScaleAbs(enh_img)
    enh_img = cv2.normalize(enh_img, enh_img, 0, 255, cv2.NORM_MINMAX)

    return enh_img


def imGamma(f, y):
    enh_img = cv2.pow(f, y)
    enh_img = cv2.convertScaleAbs(enh_img)
    enh_img = cv2.normalize(enh_img, enh_img, 0, 255, cv2.NORM_MINMAX)
    return enh_img


def imEqualHist(f):
    # Extracting the global maximum, width and height
    _, L, _, _ = cv2.minMaxLoc(f)
    M = np.array(f).shape[0]
    N = np.array(f).shape[1]

    # Transform the matrix in a row vector
    f_array = np.array(f).flatten()

    ''' USING NP HIST
    hist, _ = np.histogram(f_array, 255)
    hist = np.cumsum(hist)

    plt.plot(range(255), hist, "r--", linewidth=1)
    # plt.show()
    # plt.clf()
    '''

    hist = myHist(f_array, 255)
    hist = myCumHist(hist)
    plt.plot(range(255), hist, "r--", linewidth=1)
    #plt.show()

    for i in range(len(f_array)):
        val = int(f_array[i])
        factor = (L/(M*N))*hist[val]
        f_array[i] *= factor

    enh_img = f_array.reshape(M, N)
    enh_img = cv2.convertScaleAbs(enh_img)
    enh_img = cv2.normalize(enh_img, enh_img, 0, 255, cv2.NORM_MINMAX)

    return enh_img

def myHist(array, n_bins):
    min = np.min(array)
    max = np.max(array)

    hist = np.zeros(n_bins)

    for element in array:
        idx = math.floor(element)
        hist[idx] += 1

    return hist

def myCumHist(array):
    return np.cumsum(array)

# def imSharp(f,y):
# def imHistogram(f):
# def showHistogram(h):

enhance('arara.jpg', 1.25, 0.7, 0.3, 1)
#enhance('nap.jpg', 1.25, 0.7, 0.3, 1)
#enhance('cameraman.png', 1.25, 0.7, 0.3, 1)