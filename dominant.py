#Needs to moved to file functions.py


import cv2
import numpy as np


def create_bar(height, width, color,image):
    bar = np.ones((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return bar, (red, green, blue)


def get_dominant(image):

    img = image
    # img = cv2.imread('F:/Year 2022/Test/color_panel/r_5Q2A0028.JPG')
    height, width, _ = np.shape(img)
    background = np.ones((70,610, 3), dtype = np.uint8)
    background = 255*background


    data = np.reshape(img, (height * width, 3))
    data = np.float32(data)

    number_clusters = 40
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data, number_clusters, None, criteria, 10, flags)
    # print(centers)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bars = []
    rgb_values = []

    for index, row in enumerate(centers):
        bar, rgb = create_bar(60,15, row, img)
        bars.append(bar)
        rgb_values.append(rgb)

    img_bar = np.hstack(bars)
    x_offset = y_offset = 5
#    print(background.shape)
    background[y_offset:y_offset + img_bar.shape[0], x_offset:x_offset+img_bar.shape[1]] = img_bar

    # cv2.imshow('color',background)
    # cv2.imshow('color_2',img)
    # cv2.waitKey()


    return background






# cv2.imshow('Image', img)


# cv2.imshow('Dominant colors', img_bar)




# # cv2.imwrite('output/bar.jpg', img_bar)

# cv2.waitKey(0)