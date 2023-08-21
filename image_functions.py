import time
from PIL import Image, ImageStat
import cv2 as cv2
import numpy as np
import math
import sys
import os
import matplotlib.pyplot as plt
from skimage.morphology import rectangle
import skimage.filters as filters
from skimage import img_as_ubyte

from skimage import io,img_as_float,transform
from skimage.restoration import denoise_nl_means,estimate_sigma
from skimage.util import img_as_ubyte
from scipy import ndimage
from scipy.ndimage.filters import convolve
from numpy.linalg import norm
from segment_class import *


def set_brightness_down(img,a):     

    org_brightness = get_brightness(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    #print(hsv[...,2].max())
    if hsv[...,2].min()  < 255:
        lim = 0 
        v[v < lim] = 0
        v[v >= lim] -= 30

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    #print(org_brightness, get_brightness(img))
    return img

def desaturate(img,a):  

    # org_brightness = get_brightness(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # print(hsv[...,2].max())
    if hsv[...,1].min() < 15:
        lim = 0 
        v[v < lim] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    # print(org_brightness, get_brightness(img))
    return img

    #print('XXXXXXXXXXXXXXXX',a)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    img_mod = img_lab.copy()
    img_mod[:, :, 0] = (a * img_mod[:, :, 0]).astype(np.uint8)

    img_mod = cv2.cvtColor(img_mod, cv2.COLOR_Lab2BGR)

    return img_mod

def DarkChannel(img,sz):
    b,g,r = cv2.split(img)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    #print(dark)
    return dark

def AtmLight(img,dark):
    [h,w] = img.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = img.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(img,A,sz):
    omega = 0.5;
    im3 = np.empty(img.shape,img.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = img[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(img,p,r,eps):
    mean_I = cv2.boxFilter(img,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(img*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;
    mean_II = cv2.boxFilter(img*img,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*img + mean_b;
    return q;

def TransmissionRefine(img,et):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(img,t,A,tx = 0.1):
    res = np.empty(img.shape,img.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (img[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

def resize_file(fn, ratio):
            #   if  image.image_gs.shape[0] > image.image_gs.shape[1]: 
        #       image.image_gs = cv2.rotate(image.image_gs, cv2.cv2.ROTATE_90_CLOCKWISE)

    img = cv2.imread(fn)
    #print(fn)
    r_s = img
    #print(r_s.shape)
    aspect_ratio = r_s.shape[0]/r_s.shape[1]
    #print(r_s.shape,"f")
    if r_s.shape[1] > r_s.shape[0]:
        p = "Landscape"
    else:
        p = "Portrait"
    scale_percent = ratio

    rs_height = int(r_s.shape[0] * scale_percent / 100)
    rs_width = int(r_s.shape[1] * scale_percent / 100)          

    rs_dim = (rs_width, rs_height)
    r_src = cv2.resize(r_s,rs_dim, interpolation = cv2.INTER_AREA).astype(np.uint16)

    return (r_src)




def resize_image(img, scale_percent):
        #   if  image.image_gs.shape[0] > image.image_gs.shape[1]: 
    #       image.image_gs = cv2.rotate(image.image_gs, cv2.cv2.ROTATE_90_CLOCKWISE)
    r_s = img
    # cv2.imshow('resize',r_s)
    # cv2.waitKey()
    aspect_ratio = r_s.shape[0]/r_s.shape[1]
    #print(r_s.shape,"i")
    if r_s.shape[1] > r_s.shape[0]:
        p = "Landscape"
    else:
        p = "Portrait"

    rs_height = int(r_s.shape[0] * scale_percent / 100)
    rs_width = int(r_s.shape[1] * scale_percent / 100)          

    rs_dim = (rs_width, rs_height)
    print(rs_dim)
    r_src = cv2.resize(r_s,rs_dim, interpolation = cv2.INTER_AREA).astype(np.uint8)

    # cv2.imshow('after resize',r_src)
    # cv2.waitKey()



    return (r_src,p)

def image_grayscale(img):
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)

def convert_PIL2CV(img):
    
    cv_image = np.array(img) 
    # Convert RGB to BGR 
    cv_image = cv_image[:, :, ::-1].copy() 
    return cv_image

def edge_tracking(img, weak, strong=255):

    M, N = img.shape

    for i in range(1, M-1):
        for j in range(1, N-1):

            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def double_thresholding(Z, lowThresholdRatio=0.09, highThresholdRatio=0.17, w_p_v=75):

    highThreshold = Z.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = Z.shape
    result = np.zeros((M,N), dtype=np.int32)

    weak = np.int32(w_p_v)
    strong = np.int32(255)

    strong_i, strong_j = np.where(Z >= highThreshold)
    zeros_i, zeros_j = np.where(Z < lowThreshold)

    weak_i, weak_j = np.where((Z <= highThreshold) & (Z >= lowThreshold))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return (result, weak, strong)

def gradient_calculation(img):


    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)


    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255


    theta = np.arctan2(Iy, Ix)

    return (G, theta)

def non_maximum_suppression(G, theta):

    M, N = G.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180


    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255

               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = G[i, j+1]
                    r = G[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = G[i+1, j-1]
                    r = G[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = G[i+1, j]
                    r = G[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = G[i-1, j-1]
                    r = G[i+1, j+1]

                if (G[i,j] >= q) and (G[i,j] >= r):
                    Z[i,j] = G[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    return Z

def gaussian_blur(size=5, sigma=1.4):

    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


