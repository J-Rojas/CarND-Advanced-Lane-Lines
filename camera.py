import numpy as np
import cv2
import matplotlib.image as mpimg

class Camera:

    def __init__(self, pattern, laneProjection=None):
        self.calibrationInfo = None
        self.pattern = pattern
        self.projectionPoints = np.float32(laneProjection)

    def calibrate(self, arrImgs):

        arrCorners = []
        arrObjectPoints = []
        retArr = []
        img_size = None

        # gather chessboard corners in distorted image space, and object points
        for img in arrImgs:
            retval, corners = cv2.findChessboardCorners(img, self.pattern)
            if retval == True:
                arrCorners.append(corners)
                objp = np.zeros((self.pattern[0] * self.pattern[1], 3), np.float32)
                objp[:,:2] = np.mgrid[0:self.pattern[0],0:self.pattern[1]].T.reshape(-1, 2)
                arrObjectPoints.append(objp)
            retArr.append((img, corners))

            img_size = img.shape[0:2][::-1]

        # calibrate camera
        cM=None
        dist=None
        rvecs=None
        tvecs=None
        retval, cM, dist, rvecs, tvecs = cv2.calibrateCamera(arrObjectPoints, arrCorners, img_size, cM, dist, rvecs, tvecs)
        self.calibrationInfo = (cM, dist, rvecs, tvecs)

        return retArr

    def drawCorners(self, img, corners):
        return cv2.drawChessboardCorners(img, self.pattern, corners, corners != None)

    def undistort(self, img):
        return cv2.undistort(img, self.calibrationInfo[0], self.calibrationInfo[1], None, self.calibrationInfo[0])

    def unproject(self, img, laneWidth, imgSize, invert=False):

        #assumes the projectionPoints is compatible with the img
        w2 = int(imgSize[0] / 2)
        lane2 = laneWidth / 2
        h = imgSize[1]

        dest = np.float32([
            (w2 - lane2, 0),
            (w2 + lane2, 0),
            (w2 + lane2, h),
            (w2 - lane2, h)
        ])

        if not(invert):
            M = cv2.getPerspectiveTransform(self.projectionPoints, dest)
        else:
            M = cv2.getPerspectiveTransform(dest, self.projectionPoints)

        return cv2.warpPerspective(img, M, imgSize, flags=cv2.INTER_LINEAR)
