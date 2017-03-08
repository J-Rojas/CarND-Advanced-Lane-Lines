import numpy as np
import cv2
import scipy as sci

class Lane:

    def __init__(self):
        self.lineFit = None
        self.radius = None
        self.centerOffset = None
        self.dataPoints = None

class LaneRenderer:

    def __init__(self, laneWidthPixels, metersPerPixelX, metersPerPixelY):
        self.laneWidth = laneWidthPixels
        self.metersPerPixelX = metersPerPixelX
        self.metersPerPixelY = metersPerPixelY

    def extractPoints(self, img):
        y, x = np.where(img != 0)
        values = img != 0
        w = img[values]
        return (y, x, w)

    def findLineFit(self, tup):
        y, x, w = tup
        return np.polyfit(y,x,3,w=w**4)

    def findObjectSpaceLineFit(self, tup):
        y, x, w = tup

        # Scale the lane points wrt real world lane dimensions
        curveData = (y * self.metersPerPixelY, x * self.metersPerPixelX, w)
        actualFit = self.findLineFit(curveData)

        return actualFit

    def findLineCurvature(self, actualFit, yRef):
        # Calculate the new radii of curvature
        dy1 = np.polyder(actualFit)
        dy2 = np.polyder(actualFit, 2)
        yScaledRef = yRef * self.metersPerPixelY

        curverad = ((1 + (np.polyval(dy1, yScaledRef)**2))**1.5) / np.absolute(np.polyval(dy2, yScaledRef))
        return curverad

    def findObjectSpaceX(self, actualFit, yRef):
        yScaledRef = yRef * self.metersPerPixelY
        return np.polyval(actualFit, yScaledRef)

    def findLaneFit(self, imgLeft, imgRight):

        left = self.extractPoints(imgLeft)
        right = self.extractPoints(imgRight)

        leftFit = self.findLineFit(left)
        rightFit = self.findLineFit(right)

        # adjust the lane fit so that both lines are parallel, using the line with the longest trail of points
        ylmin = np.min(left[0])
        yrmin = np.min(right[0])

        yadj, xadj, wadj = left if ylmin > yrmin else right
        yadjr = (yrmin, ylmin, -self.laneWidth) if ylmin > yrmin else (ylmin, yrmin, self.laneWidth)
        imgCp = imgRight if ylmin > yrmin else imgLeft
        bestFit = rightFit if ylmin > yrmin else leftFit

        #adjust the line fit by copy data points within the adjustment range
        if yadjr[1] - yadjr[0] > 50:

            print('adding extra points: diff = ', yadjr[1] - yadjr[0])

            delta = yadjr[2]
            absdelta = abs(yadjr[2])
            deriv = np.polyder(bestFit)

            subimg = imgCp[yadjr[0]:yadjr[1]-50,:]

            ycp, xcp, wcp = self.extractPoints(subimg)
            ycp += yadjr[0]

            # translate the points perpendicular to the tanget of the curve at the given point
            yn, xn = [], []
            for yo, xo in zip(ycp, xcp):
                dy = np.polyval(deriv, yo)
                dx = np.sqrt(1 - dy*dy)
                if delta < 0:
                    dx = -dx

                yn.append(int(yo + dy * absdelta))
                xn.append(int(xo + dx * absdelta))

            print(np.dstack((xn, yn)))

            xadj = np.concatenate((xadj, np.array(xn)))
            yadj = np.concatenate((yadj, np.array(yn)))
            wadj = np.concatenate((wadj, wcp))

            if ylmin > yrmin:
                left = (yadj, xadj, wadj)
                leftFit = self.findLineFit(left)
            else:
                right = (yadj, xadj, wadj)
                rightFit = self.findLineFit(right)

        # find scaled line fit in object space
        leftObjectFit = self.findObjectSpaceLineFit(left)
        rightObjectFit = self.findObjectSpaceLineFit(right)

        # find curvature radius
        leftRadius = self.findLineCurvature(leftObjectFit, imgLeft.shape[0])
        rightRadius = self.findLineCurvature(rightObjectFit, imgRight.shape[0])

        print(leftRadius, rightRadius)
        radius = leftRadius if ylmin < yrmin else rightRadius

        #find lane offset
        lObjectLanePosition = self.findObjectSpaceX(leftObjectFit, imgLeft.shape[0])
        rObjectLanePosition = self.findObjectSpaceX(rightObjectFit, imgRight.shape[0])
        laneObjectCenter = (lObjectLanePosition + rObjectLanePosition) / 2

        print(lObjectLanePosition, rObjectLanePosition, laneObjectCenter)

        offset = (max(imgLeft.shape[1], imgLeft.shape[1]) / 2) * self.metersPerPixelX - laneObjectCenter

        lane = Lane()
        lane.lineFit = (leftFit, rightFit)
        lane.dataPoints = (left, right)
        lane.radius = radius
        lane.centerOffset = offset

        return lane

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
