"""
Script performing image processing
"""

import collect
import network
import cv2
import numpy as np
from PIL import Image

def init_nn():
    """
    Initializes and load pre-trained models of the neural network
    """

    try:
        net = network.NeuralNetwork()
        net.load()

    except IOError:
        training_data, validation_data, test_data = collect.load_mnist()

        net = network.NeuralNetwork(sizes=[784, 100, 10])
        net.fit(training_data, validation_data)

        net.save()

    return net

def np_empty(np_list):
    """
    Checks if elements in np list are None or not
    Returns True if empty, else False
    """

    for r in np_list:
        if r:
            return False
    return True

def get_decimal_in_box(input_im):
    """
    Returns decimal extracted from the image

    Params :
        - input_im : image containing digits with/without decimals
    """

    input_im = Image.open('img/photo_2.jpg')

    net = init_nn()

    im = np.array(input_im)

    cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    im_gray = cv2.bilateralFilter(im_gray, 11, 17, 17)
    im_gray = cv2.bilateralFilter(im_gray, 3, 5, 5)
    im_gray = cv2.GaussianBlur(im_gray, (3, 5), 0,3)

    im_th = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 7, 1)

    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))

    digits = []
    min_rect_area = 90

    # For each rectangular region, feed the image into NN and store predictions
    for rect in rects:
        # Rectangles less than area of size min_rect_area are excluded
        if rect[2]*rect[3] < min_rect_area:
            continue

        # Draw the rectangles
        cv2.rectangle(
            im,
            (rect[0], rect[1]),
            (rect[0] + rect[2], rect[1] + rect[3]),
            (0, 255, 0),
            3
        )

        # Make the rectangular region around the digit
        leng = int(rect[3] * 0.8) ### change this value
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)

        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        roi_list = roi.tolist()

        # skipping empty boxes
        if np_empty(roi_list):
            continue

        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, kernel, iterations=1)

        roi_hog = np.reshape(roi, (784,1))
        nbr = net.predict(roi_hog)

        digits.append((int(nbr), rect))

        cv2.putText(
            im,
            str(int(nbr)),
            (rect[0], rect[1]),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 0, 0),
            1
        )

    cv2.imshow("Resulting Image with Rectangular ROIs", im)
    cv2.waitKey()

    return digits
