import numpy as np
import cv2
import os
import sys
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from camera import Camera
from linefilter import LineFilter
from linefinder import LineFinder
from lanerenderer import LaneRenderer

projectionPoints = [(620, 435), (658, 433), (1063, 694), (227, 705)]
camera = Camera((9,6), projectionPoints)

camimgs = glob.glob('./camera_cal/calibration*.jpg')
camera_images = []
for imgPath in camimgs:
    camera_images.append(mpimg.imread(imgPath))
camera_calibrations = camera.calibrate(camera_images)

undistorted_imgs = []
distorted_imgs = []
for img, corners in camera_calibrations:
    distorted_imgs.append(img)
    undistorted_imgs.append(camera.undistort(camera.drawCorners(img, corners)))

file1 = sys.argv[1]
file2 = sys.argv[2]

video = cv2.VideoCapture()
video.open(file1)
video_out = cv2.VideoWriter()
video_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

print(video_size, video.get(cv2.CAP_PROP_FOURCC))

if os.path.isfile(file2):
    os.remove(file2)
video_out.open(
    file2,
    cv2.VideoWriter_fourcc(*'MP42'),
    video.get(cv2.CAP_PROP_FPS),
    video_size
)

lf=LineFilter()
linefinder = LineFinder(75, 50, 10, 40, 40, 15, 200, 350)
lanewidthPixels = 100
dashLaneLineLengthPixels = 30
lanerender=LaneRenderer(lanewidthPixels, 3.7/lanewidthPixels, 3./dashLaneLineLengthPixels)
frame_history = []
lane_history = []
lane_details = { "radius": 0, "offset": 0 }

MAX_FRAME_HISTORY = 5
MAX_LANE_HISTORY = 10

def drawText(img, region_w, region_w2, text, line):
    textSize, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)    
    cv2.putText(img, text, (int((region_w + region_w2 - textSize[0]) / 2), (5 + textSize[1]) * line), cv2.FONT_HERSHEY_SIMPLEX, 1, (255., 255., 255.), 2)

def pipeline(video, frame_count):

    retval, nextframe = video.read()

    if not(retval):
        return False

    try:
        nextframe = cv2.cvtColor(nextframe, cv2.COLOR_BGR2RGB)
        img = camera.undistort(nextframe)
        un_test_image = img
        hsl_test_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hsv_test_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        kernelSize=7
        magnitudeThreshold=(10.0, 20.1)
        angleThreshold=(0.7,1.5)
        whiteLines=()
        H = 0
        S = 2
        L = 1
        laneWidth=100
        srcImg = hsl_test_image

        img1 = [lf.source(hsv_test_image).
            colorThreshold(H, (0, 30), kernel=5).
            colorThreshold(1, (100, 255), kernel=5).
            colorThreshold(2, (130, 255), kernel=5).
            sobelMagThreshold(S, (0.5, 255), 5).
            markLines()
        ]

        img1_1 = [lf.source(hsv_test_image).
            colorThreshold(H, (0, 30), kernel=5).
            colorThreshold(1, (0, 30), kernel=5).
            colorThreshold(2, (200, 255), kernel=5).
            sobelMagThreshold(S, (1, 255), 5).
            markLines()
        ]

        img1 = cv2.bitwise_or(np.array(img1), np.array(img1_1))

        #shadow resistant filter
        #img2 = [lf.source(srcImg).
        #    colorThreshold(0, (15, 36)).
        #    colorThreshold(S, (60, 255)).
        #    sobelXThreshold(L, (7, 255), 5).
        #    sobelYThreshold(L, (7, 255), 5).
        #    markLines()
        #       ]

        img2 = [lf.source(hsv_test_image).
            colorThreshold(H, (0, 30), kernel=9).
            colorThreshold(1, (70, 160), kernel=9).
            sobelMagThreshold(1, (2, 255), 5).
            colorThreshold(2, (100, 255), kernel=5).
            sobelMagThreshold(2, (1.5, 255), 5).
            markLines()
       ]

        #lightness
        img3 = [lf.source(hsv_test_image).
            colorThreshold(2, (170, 255), kernel=3).
            sobelMagThreshold(2, (2, 255), 3).
            markLines()
        ]

        #saturation
        img3_1 = [lf.source(hsv_test_image).
            colorThreshold(1, (0, 20), kernel=3).
            #sobelMagThreshold(S, (1, 255), 3).
            markLines()
        ]

        img3 = cv2.bitwise_and(np.array(img3), np.array(img3_1))


        marked = [
            lf.combine([img1[0], img2[0], img3[0]], useColors=np.array([[0, 1, 0], [0, 0, 1], [1,0,0]]))
        ]

        filtered_image = np.uint8(marked[0] * 255)

        overhead_image_color = camera.unproject(marked[0], laneWidth, (1280, 720))
        if len(frame_history) >= MAX_FRAME_HISTORY:
            frame_history.pop(0)
        frame_history.append(overhead_image_color)

        weights = np.logspace(-len(frame_history), 0, num=len(frame_history), base=2) if len(frame_history) > 1 else [1.0]

        print(weights)

        overhead_image_color_history = np.array([])
        overhead_image_color_history.resize(overhead_image_color.shape)

        history = np.array(frame_history)

        r_channel = np.average(history[:,:,:,0], weights=weights, axis=0)
        g_channel = np.average(history[:,:,:,1], weights=weights, axis=0)
        b_channel = np.average(history[:,:,:,2], weights=weights, axis=0)

        print(overhead_image_color.shape, overhead_image_color_history.shape, r_channel.shape)

        overhead_image_color_history[:,:,0] = r_channel
        overhead_image_color_history[:,:,1] = g_channel
        overhead_image_color_history[:,:,2] = b_channel
        overhead_image_merged = np.sum(overhead_image_color_history, axis=2)

        left_lane_img, right_lane_img = linefinder.maskLines(overhead_image_merged, drawBoundaries=False)

        lane = lanerender.findLaneFit(left_lane_img, right_lane_img)
        l_fit, r_fit = lane.lineFit
        l_min, r_min = lane.minY

        l_points = lanerender.generateLinePoints(left_lane_img.shape, l_fit, 5, minY=l_min)
        r_points = lanerender.generateLinePoints(right_lane_img.shape, r_fit, 5, minY=r_min)

        left_line_img = lanerender.renderLinePoints(l_points,
          (left_lane_img.shape[0], left_lane_img.shape[1], 3),
          thickness=3, alpha=1.0, color=(0, 0, 255)
        )
        right_line_img = lanerender.renderLinePoints(r_points,
          (right_lane_img.shape[0], right_lane_img.shape[1], 3),
          thickness=3, alpha=1.0, color=(255, 0, 0)
        )

        merged_lane_line_img = cv2.bitwise_or(left_line_img, right_line_img)

        r = r_points[::-1]
        center_points = np.concatenate((np.array(l_points),
                                       np.array(r)), axis=0
                      )
        center_lane_img = lanerender.renderLinePoints(center_points,
          (left_lane_img.shape[0], left_lane_img.shape[1], 3), color=(0.,255.,0.), thickness=3, closed=True
        )

        center_projected_lane_line_img = camera.unproject(center_lane_img, laneWidth, (1280, 720), invert=True)

        combined_projected_lane_img = center_projected_lane_line_img
        overlay_lane_img = cv2.addWeighted(un_test_image, 1, combined_projected_lane_img, 0.25, 0.0) # overlay the orignal road image with window results
        combined_lane_line_img = left_line_img + right_line_img
        projected_lane_line_img = camera.unproject(combined_lane_line_img, laneWidth, (1280, 720), invert=True)
        overlay_lane_img = cv2.addWeighted(overlay_lane_img, 1, projected_lane_line_img, 1.0, 0.0)

        final_img = overlay_lane_img.copy()
        region_h = int(final_img.shape[0]/2.5)
        region_w = int(final_img.shape[1]/2.5)
        region_w2 = int(region_w * 1.5)
        dest1 = cv2.resize(filtered_image, (region_w, region_h), interpolation=cv2.INTER_AREA)
        dest2 = cv2.resize(merged_lane_line_img, (region_w, region_h), interpolation=cv2.INTER_AREA)

        final_img[0:region_h,0:region_w] = dest1
        final_img[0:region_h,region_w2:] = dest2

        # store lane history
        if len(lane_history) > MAX_LANE_HISTORY:
            lane_history.pop(0)
        lane_history.append(lane)

        # average lane radius and offset
        radius = lane_details["radius"]
        offset = lane_details["offset"]

        if frame_count % 10 == 0:
            total_radius, total_offset = 0, 0
            for item in lane_history:
                total_radius += item.radius
                total_offset += item.centerOffset
            lane_details["radius"] = radius = total_radius / len(lane_history)
            lane_details["offset"] = offset = total_offset / len(lane_history)

        #draw curvature and offset
        textSize, baseLine = cv2.getTextSize("Curvature Radius", cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_h = textSize[1] + 5

        cv2.rectangle(final_img, (region_w, 0), (region_w2, text_h * 5 + 5), (0., 0., 0.), cv2.FILLED)

        drawText(final_img, region_w, region_w2, "Curvature Radius", 1)
        drawText(final_img, region_w, region_w2, "{:.1f} m".format(radius), 2)
        drawText(final_img, region_w, region_w2, "Lane Offset", 4)
        drawText(final_img, region_w, region_w2, "{:.2f} m".format(offset), 5)

        return final_img

    except ValueError:
        mpimg.imsave(nextframe, './error_img.jpg')
        return False

i = 0
while True:
    frame = pipeline(video, i)
    if frame is not False:
        print('writing frame ', i)
        video_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    else:
        break
    i+=1
