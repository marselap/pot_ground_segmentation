#!/usr/bin/env python3

import numpy as np
import cv2 as cv


class Segmentor():
    def __init__(self):
        self.image = None

        self.mask_lower = (17, 10, 0)
        self.mask_upper = (100, 100, 65)

        self.min_contour_size = 100

        self.kernel = 5

        pass

    def new_image(self, image):
        self.image = image

    def color_filter(self):

        src = self.image
        hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

        if self.mask_lower[0] > self.mask_upper[0]:
            tmp_low = (0, self.mask_lower[1], self.mask_lower[2])
            tmp_upp = (255, self.mask_upper[1], self.mask_upper[2])
            lower_range = (tmp_low, self.mask_lower)
            upper_range = (self.mask_upper, tmp_upp)
        
        else:
            lower_range = [self.mask_lower]
            upper_range = [self.mask_upper]
        

        masks = []
        for (low, upp) in zip(lower_range, upper_range):
            masks.append(cv.inRange(hsv, low, upp))
        
        mask_tmp = masks[0]
        for mask in masks:
            mask_tmp = cv.bitwise_or(mask_tmp, mask)

        result = cv.bitwise_and(src, src, mask = mask_tmp)
        return result, cv.cvtColor(mask_tmp, cv.COLOR_GRAY2RGB)
        # return result
        # # lower_range = np.array([0, 0, 10])
        # # upper_range = np.array([100, 100, 250])
        # mask = cv.inRange(hsv, lower_black, upper_black)
        # result = cv.bitwise_and(src, src, mask = mask)
        # return result


    def segment(self):

        src = self.image

        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3,3))

        threshold = 50



        masked, mask = self.color_filter()

        kernel = np.ones((self.kernel,self.kernel),np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        canny_output = cv.Canny(mask, threshold, threshold * 2)


        contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

        # for i in range(len(contours)):
        #     cv.drawContours(drawing, contours, i, (0,255,0), 2, cv.LINE_8, hierarchy, 0)

        for cnt in contours:
            if cnt.size > self.min_contour_size:
                cv.drawContours(drawing, [cnt], 0, 255, -1)
                # cv.drawContours(drawing, [cnt], 0, (0,255,0), 2, cv.LINE_8)


        self.image = None
        #do image processing...
        return masked, mask, drawing
        return mask
        return result


