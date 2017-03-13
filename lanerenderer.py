import numpy as np
import cv2
import scipy as sci

class Lane:

    def __init__(self):
        self.lineFit = None
        self.radius = None
        self.centerOffset = None
        self.dataPoints = None
        self.minY = None

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
        return np.polyfit(y,x,3,w=w**2+y)

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

    def findLaneFit(self, imgLeft, imgRight, updateImages=False):

        left = self.extractPoints(imgLeft)
        right = self.extractPoints(imgRight)

        leftFit = self.findLineFit(left)
        rightFit = self.findLineFit(right)

        # adjust the lane fit so that both lines are parallel, using the line with the longest trail of points
        ylmin = np.min(left[0])
        yrmin = np.min(right[0])

        def yLaneWidthAdjust(y):
            maxsize = 3 * imgLeft.shape[0] / 4
            ymin = min(y, maxsize)
            return (maxsize - y) / maxsize * self.laneWidth / 3 + self.laneWidth

        yadj, xadj, wadj = left if ylmin > yrmin else right
        yadjr = (yrmin, ylmin, -yLaneWidthAdjust(yrmin)) if ylmin > yrmin else (ylmin, yrmin, yLaneWidthAdjust(ylmin))
        imgCp = imgRight if ylmin > yrmin else imgLeft
        bestFit = rightFit if ylmin > yrmin else leftFit
        leastFit = leftFit if ylmin > yrmin else rightFit

        #adjust the line fit by coping data points within the adjustment range
        if yadjr[1] - yadjr[0] > 50:

            #print('adding extra points: diff = ', yadjr[1] - yadjr[0])

            delta = yadjr[2]
            absdelta = abs(yadjr[2])
            deriv = np.polyder(bestFit)

            subimg = imgCp[yadjr[0]:yadjr[1]-50,:]

            ycp, xcp, wcp = self.extractPoints(subimg)
            ycp += yadjr[0]

            # determine if the fit curvature is near zero, if so, reflect the augemented data points across the central lane line
            x1 = np.polyval(bestFit, yadjr[0])
            x2 = np.polyval(bestFit, (imgCp.shape[0] + yadjr[0]) / 2)
            x3 = np.polyval(bestFit, imgCp.shape[0])
            dx1 = np.polyval(deriv, yadjr[0])
            dx2 = np.polyval(deriv, imgCp.shape[0])

            print(x1 - x2, x2 - x3, (x1 - x2) - (x2 - x3), abs(dx1 - dx2))

            if abs((x1 - x2) - (x2 - x3)) < 16 and abs(dx1 - dx2) < 0.06:
                print('curvature is near zero')

                # reflect points
                xlmin = np.polyval(leastFit, yadjr[1])
                xlmax = np.polyval(leastFit, imgCp.shape[0])

                xbmin = np.polyval(bestFit, yadjr[1])
                xbmax = np.polyval(bestFit, imgCp.shape[0])

                print(yadjr)

                m1 = (xlmax - xlmin)/(imgCp.shape[0] - yadjr[1])
                m2 = (xbmax - xbmin)/(imgCp.shape[0] - yadjr[1])
                m = (m1 + m2) / 2

                print(m1, m2, m)

                l = np.array([m, 1])
                laneorigin = np.array([(xbmax + xlmax) / 2, imgCp.shape[0]])

                #print(laneorigin)

                vcp = (np.dstack((xcp, ycp)) - laneorigin).reshape((-1,2))

                #print(vcp)

                vldot = np.dot(vcp, l)
                lldot = np.dot(l,l)
                scdot = (vldot / lldot).reshape(-1,1)

                #print(vldot, lldot, scdot)

                rcp = 2 * scdot * l - vcp

                vvcp = rcp + laneorigin

                #print(vvcp)

                xn = np.int32(vvcp[:,0])
                yn = np.int32(vvcp[:,1])

                #print(xn, yn)

                #print (vvcp)

                #xd = xcp - xmmax
                #xn = np.array(xcp - (xd * 2), dtype=np.int32)
                #yn = ycp

                xadj = np.concatenate((xadj, xn))
                yadj = np.concatenate((yadj, yn))
                wadj = np.concatenate((wadj, wcp))

            else:
                # the line is curved; model the missing line by translating the points from the known line
                # to a region perpendicular to the tanget of the curve at the given point
                yn, xn = [], []
                for yo, xo in zip(ycp, xcp):
                    dy = np.polyval(deriv, yo)
                    dx = np.sqrt(1 - dy*dy)
                    if delta < 0:
                        dx = -dx

                    yn.append(int(yo + dy * absdelta))
                    xn.append(int(xo + dx * absdelta))

                #print(np.dstack((xn, yn)))

                xadj = np.concatenate((xadj, np.array(xn)))
                yadj = np.concatenate((yadj, np.array(yn)))
                wadj = np.concatenate((wadj, wcp))

            if ylmin > yrmin:
                left = (yadj, xadj, wadj)
                leftFit = self.findLineFit(left)
                if updateImages:
                    imgLeft[yn,xn] = wcp
            else:
                right = (yadj, xadj, wadj)
                rightFit = self.findLineFit(right)
                if updateImages:
                    imgRight[yn,xn] = wcp

        # find scaled line fit in object space
        leftObjectFit = self.findObjectSpaceLineFit(left)
        rightObjectFit = self.findObjectSpaceLineFit(right)

        # find curvature radius
        leftRadius = self.findLineCurvature(leftObjectFit, imgLeft.shape[0])
        rightRadius = self.findLineCurvature(rightObjectFit, imgRight.shape[0])

        print(leftRadius, rightRadius)

        #XXX use left radius since the left lane line is more stable in the project demo video
        radius = leftRadius # if ylmin < yrmin else rightRadius

        #find lane offset
        lObjectLanePosition = self.findObjectSpaceX(leftObjectFit, imgLeft.shape[0])
        rObjectLanePosition = self.findObjectSpaceX(rightObjectFit, imgRight.shape[0])
        laneObjectCenter = (lObjectLanePosition + rObjectLanePosition) / 2

        #print(lObjectLanePosition, rObjectLanePosition, laneObjectCenter)

        offset = (max(imgLeft.shape[1], imgLeft.shape[1]) / 2) * self.metersPerPixelX - laneObjectCenter

        lane = Lane()
        lane.lineFit = (leftFit, rightFit)
        lane.dataPoints = (left, right)
        lane.radius = radius
        lane.centerOffset = offset
        lane.minY = (np.min(left[0]), np.min(right[0]))

        return lane

    def generateLinePoints(self, shape, fit, step, minY=0):

        points = []

        for y in np.arange(minY, shape[0]+1, step):
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
