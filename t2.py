# Nome: Luis Paulo Falchi Justino
# NUSP: 8937479

import cv2
import numpy as np
import sys

def enhance(filename, gamma, a, b, imshow):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Convert the image to 32bit float
    imgf = np.float32(img)

    # List of enhanced images
    enhanced_images = []

    # Keep the original image for comparison sakes
    enhanced_images.append(img)

    # Applying different enhancement methods
    log_enhanced = imLog(imgf)
    enhanced_images.append(log_enhanced)

    gamma_enhanced = imGamma(imgf, gamma)
    enhanced_images.append(log_enhanced)

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
    enh_img = cv2.exp(f, y)
    enh_img = cv2.normalize(enh_img, enh_img, 0, 255, cv2.NORM_MINMAX)

    return enh_img

# def imEqualHist(f):
# def imSharp(f,y):
# def imHistogram(f):
# def showHistogram(h):

enhance('arara.jpg', 1.25, 0.7, 0.3, 1)
#enhance('nap.jpg', 1.25, 0.7, 0.3, 1)
#enhance('cameraman.png', 1.25, 0.7, 0.3, 1)