import numpy as np
import cv2
import math
from scipy import signal

class LineFinder:

    def __init__(self, laneWidth, filterSize, window_width, window_height, margin, terminate):
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.filterSize = filterSize
        self.laneWidth = laneWidth
        self.terminate = terminate if terminate else math.inf

    def copy_mask(self, imgInput, imgOutput, width, height, center,level):
        shape = imgInput.shape
        self.set_mask(imgInput[
                int(shape[0]-(level+1)*height):int(shape[0]-level*height),
                max(0,int(center-width/2)):min(int(center+width/2),shape[1])
            ], imgOutput, width, height, center, level)

    def set_mask(self, value, imgOutput, width, height, center,level):
        shape = imgOutput.shape
        imgOutput[
            int(shape[0]-(level+1)*height):int(shape[0]-level*height),
            max(0,int(center-width/2)):min(int(center+width/2),shape[1])
        ] = value

    def drawWeightedLayer(self, img, layer):
        output = np.array(img * 255 / np.max(img), np.uint8)
        zero_channel = np.zeros_like(output) # create a zero color channle
        template = np.array(cv2.merge((zero_channel,np.uint8(layer * 255),zero_channel)),np.uint8) # make window pixels green
        output = np.array(cv2.merge((output,output,output)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(output, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

        return output

    def find_window_centroids(self, image, window_width, window_height, margin):

        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(self.filterSize) # Create our window template that we will use for convolutions
        window2d = np.ones((self.filterSize,self.filterSize)) # Create our window template that we will use for convolutions
        shape = image.shape

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        hw = int(shape[1]/2)
        lw = self.laneWidth
        l_start_layer = image[int(7*shape[0]/8):,hw-lw:hw]
        conv_signal = np.sum(signal.convolve2d(l_start_layer, window2d, mode='same'), axis=0)
        l_center = np.argmax(conv_signal)+(hw-lw)

        r_start_layer = image[int(7*shape[0]/8):,hw:hw+lw]
        conv_signal = np.sum(signal.convolve2d(r_start_layer, window2d, mode='same'), axis=0)
        r_center = np.argmax(conv_signal)+hw

        # Add what we found for the first layer
        window_centroids.append((l_center,r_center,window_width,window_width))
        offset = window_width/2

        lMissing = 0
        rMissing = 0

        diffCentroid = (0, 0)

        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(shape[0]/window_height)):

            # convolve the window into the vertical slice of the image
            #image_layer = np.sum(image[int(shape[0]-(level+1)*window_height):int(shape[0]-level*window_height),:], axis=0)
            image_layer = image[int(shape[0]-(level+1)*window_height):int(shape[0]-level*window_height),:]
            #conv_signal = np.convolve(window, image_layer)
            conv_signal = np.sum(signal.convolve2d(image_layer, window2d, mode='same'), axis=0)

            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            l_min_index = int(min(max(l_center-margin+diffCentroid[0],0),shape[1]))
            l_max_index = int(max(min(l_center+margin+diffCentroid[0],shape[1]),0))

            left_values = conv_signal[l_min_index:l_max_index]
            if np.max(left_values) > 0 and lMissing <= self.terminate:
                l_center = np.argmax(left_values)+l_min_index
                lMissing = 0
            else:
                lMissing += window_height

            # Find the best right centroid by using past right center as a reference
            r_min_index = int(min(max(r_center-margin+diffCentroid[1],0),shape[1]))
            r_max_index = int(max(min(r_center+margin+diffCentroid[1],image.shape[1]),0))

            right_values = conv_signal[r_min_index:r_max_index]
            if np.max(right_values) > 0 and rMissing <= self.terminate:
                r_center = np.argmax(right_values)+r_min_index
                rMissing = 0
            else:
                rMissing += window_height

            if lMissing > 0:
                l_center += diffCentroid[1]
            if rMissing > 0:
                r_center += diffCentroid[0]

            print(diffCentroid)

            # Add what we found for that layer
            prevCentroid = None
            if len(window_centroids) > 0:
                prevCentroid = window_centroids[-1]
                diffCentroid = (l_center - prevCentroid[0], r_center - prevCentroid[1])

            window_centroids.append(
                (l_center,r_center,
                 window_width if lMissing <= self.terminate else 0,
                 window_width if rMissing <= self.terminate else 0
                )
            )

        return window_centroids

    def maskLines(self, img, drawBoundaries=False):

        window_width = self.window_width
        window_height = self.window_height
        window_centroids = self.find_window_centroids(img, window_width, window_height, self.margin)

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)

        print("centroids found: ", window_centroids)

        boundariesLeft = np.zeros_like(img) if drawBoundaries else None
        boundariesRight = np.zeros_like(img) if drawBoundaries else None

        # If we found any window centers
        if len(window_centroids) > 0:

            # Go through each level and draw the windows
            for level in range(0,len(window_centroids)):
                # Window_mask is a function to draw window areas
                centroid = window_centroids[level]
                if centroid[0] != -1:
                    self.copy_mask(img,l_points,centroid[2],window_height,centroid[0],level)
                if centroid[1] != -1:
                    self.copy_mask(img,r_points,centroid[3],window_height,centroid[1],level)
                if drawBoundaries:
                    if centroid[0] != -1:
                        self.set_mask(1, boundariesLeft, centroid[2],window_height,centroid[0],level)
                    if centroid[1] != -1:
                        self.set_mask(1, boundariesRight, centroid[3],window_height,centroid[1],level)

        # Draw the results
        if drawBoundaries:
            l_points = self.drawWeightedLayer(l_points, boundariesLeft)
            r_points = self.drawWeightedLayer(r_points, boundariesRight)

        return (l_points, r_points)
