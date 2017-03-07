import numpy as np
import cv2

class LineFilter:

    def __init__(self):
        self.img = None

    def source(self, img):
        self.img = img
        self.mask = None
        self.output = None
        return self

    def updateMask(self, mask, op):

        op = np.logical_and if op == 'and' else np.logical_or

        if self.mask != None:
            mask = op(mask, self.mask)
        self.mask = mask

    def sobelX(self, channel, kernel, bAbs=True):
        pixels = self.img[:,:,channel]

        deriv_filter = cv2.getDerivKernels(dx=1, dy=0, ksize=kernel, normalize=True)
        sobelx = cv2.sepFilter2D(pixels, cv2.CV_64F, deriv_filter[0], deriv_filter[1])

        if bAbs:
            sobelx = np.abs(sobelx)

        return sobelx

    def sobelY(self, channel, kernel, bAbs=True):
        pixels = self.img[:,:,channel]

        deriv_filter = cv2.getDerivKernels(dx=0, dy=1, ksize=kernel, normalize=True)
        sobely = cv2.sepFilter2D(pixels, cv2.CV_64F, deriv_filter[0], deriv_filter[1])

        if bAbs:
            sobely = np.abs(sobely)

        return sobely


    def colorThreshold(self, channel, threshold, op='and'):

        pixels = self.img[:,:,channel]
        mask = [(pixels >= threshold[0]) & (pixels <= threshold[1])]

        self.output = pixels
        self.updateMask(mask, op)

        return self

    def sobelXThreshold(self, channel, mag, kernel=3, op='and'):

        pixels = self.img[:,:,channel]

        deriv_filter = cv2.getDerivKernels(dx=1, dy=0, ksize=kernel, normalize=True)
        sobelx = cv2.sepFilter2D(pixels, cv2.CV_64F, deriv_filter[0], deriv_filter[1])
        abs_sobel = np.abs(sobelx)

        mask = [(abs_sobel >= mag[0]) & (abs_sobel <= mag[1])]

        self.output = abs_sobel
        self.updateMask(mask, op)

        return self

    def sobelYThreshold(self, channel, mag, kernel=3, op='and'):

        pixels = self.img[:,:,channel]

        deriv_filter = cv2.getDerivKernels(dx=0, dy=1, ksize=kernel, normalize=True)
        sobel = cv2.sepFilter2D(pixels, cv2.CV_64F, deriv_filter[0], deriv_filter[1])
        abs_sobel = np.abs(sobel)

        mask = [(abs_sobel >= mag[0]) & (abs_sobel <= mag[1])]

        self.output = abs_sobel
        self.updateMask(mask, op)

        return self

    def sobelMagThreshold(self, channel, mag, kernel=3, op='and'):

        pixels = self.img[:,:,channel]

        deriv_filter = cv2.getDerivKernels(dx=1, dy=1, ksize=kernel, normalize=True)
        sobel = cv2.sepFilter2D(pixels, cv2.CV_64F, deriv_filter[0], deriv_filter[1])
        abs_sobel = np.abs(sobel)

        mask = [(abs_sobel >= mag[0]) & (abs_sobel <= mag[1])]

        self.output = abs_sobel
        self.updateMask(mask, op)

        return self

    def sobelAngleThreshold(self, channel, angle, kernel=3, op='and'):

        pixels = self.img[:,:,channel]

        deriv_filter1 = cv2.getDerivKernels(dx=1, dy=0, ksize=kernel, normalize=True)
        deriv_filter2 = cv2.getDerivKernels(dx=0, dy=1, ksize=kernel, normalize=True)
        sobelx = cv2.sepFilter2D(pixels, cv2.CV_64F, deriv_filter1[0], deriv_filter1[1])
        sobely = cv2.sepFilter2D(pixels, cv2.CV_64F, deriv_filter2[0], deriv_filter2[1])

        sobelarctan = np.abs(np.arctan2(sobely, sobelx))
        mask = [(sobelarctan >= angle[0]) & (sobelarctan <= angle[1])]

        self.output = sobelarctan
        self.updateMask(mask, op)

        return self

    def markLines(self, binary=True):
        if binary:
            output = np.zeros_like(self.img[:,:,0])
            output[self.mask[0]]=1
        else:
            output = np.float32(self.output)
            output[np.logical_not(self.mask)[0]] = 0
            output /= output.std()

        return output

    def combine(self, imgMasks, useColors=None):

        output = np.zeros_like(self.img, dtype=np.float32)

        for img, color in zip(imgMasks, useColors):
            if color == None:
                color = np.array([1, 1, 1])
            output += np.outer(img, color).reshape(output.shape)

        return output
