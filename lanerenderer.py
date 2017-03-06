import numpy as np
import cv2
import scipy as sci

class LaneRenderer:

    def __init__(self, laneWidth):
        self.laneWidth = laneWidth

    def extractPoints(self, img):
        y, x = np.where(img != 0)
        values = img != 0
        w = img[values]
        return (y, x, w)

    def findLineFit(self, img, tup):
        y, x, w = tup
        return np.polyfit(y,x,3,w=w)

    def findLaneFit(self, imgLeft, imgRight):

        left = self.extractPoints(imgLeft)
        right = self.extractPoints(imgRight)

        print(left, right)

        # adjust the lane fit so that both lines are parallel, using the line with the longest trail of points
        ylmin = np.min(left[0])
        yrmin = np.min(right[0])

        yadj, xadj, wadj = left if ylmin > yrmin else right
        yadjr = (yrmin, ylmin, -self.laneWidth) if ylmin > yrmin else (ylmin, yrmin, self.laneWidth)
        imgCp = imgRight if ylmin > yrmin else imgLeft

        #copy data points within the adjustment range
        if yadjr[1] - yadjr[0] > 50:

            subimg = imgCp[yadjr[0]:yadjr[1]-50,:]

            ycp, xcp, wcp = self.extractPoints(subimg)

            xcp += yadjr[2]
            ycp += yadjr[0]

            print(xcp, ycp)

            xadj = np.concatenate((xadj, xcp))
            yadj = np.concatenate((yadj, ycp))
            wadj = np.concatenate((wadj, wcp))

            if ylmin > yrmin:
                left = (yadj, xadj, wadj)
            else:
                right = (yadj, xadj, wadj)

        leftFit = self.findLineFit(imgLeft, left)
        rightFit = self.findLineFit(imgRight, right)

        return (leftFit, rightFit)

    def generateLinePoints(self, shape, fit, step):

        points = []

        for y in np.arange(0, shape[0]+1, step):
            x = np.polyval(fit, y)
            points.append((x, y))

        return np.array(points)

    def drawWeightedLayer(self, img, layer, alpha):
        output = np.array(img * 255 / np.max(img), np.uint8)
        output = cv2.addWeighted(output, 1, layer, alpha, 0.0) # overlay the orignal road image with window results
        return output


    def renderLinePoints(self, points, shape, img=None, color=(1.0, 0.0, 0.0), alpha=0.5, thickness=2, closed=False):
        layer = np.zeros(shape, np.uint8)
        if closed:
            cv2.fillPoly(layer, points.reshape(1,-1,2).astype(np.int32), color)
        else:
            cv2.polylines(layer, points.reshape(1,-1,2).astype(np.int32), closed, color, thickness)
        if img is not None:
            return self.drawWeightedLayer(img, layer, alpha*255)
        return layer
