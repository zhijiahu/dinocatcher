
import argparse
import imutils
from imutils.video import VideoStream
import cv2
import time
import datetime
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-w", "--webcam", help="use webcam", action='store_true')
args = vars(ap.parse_args())

# define the color ranges
colorRanges = [
    ((29, 86, 6), (64, 255, 255), "door")]


doorOriginalPos = None
doorNewPos = None
alarmStartTime = None
threshold = 10

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    if args.get("webcam"):
        vs = cv2.VideoCapture(0)
    else:
        vs = VideoStream(usePiCamera=True).start()

    time.sleep(2.0)
# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = vs.read()

    # if we are viewing a video and we did not grab a frame, then we have
    # reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # loop over the color ranges
    for (lower, upper, colorName) in colorRanges:
	# construct a mask for all colors in the current HSV range, then
	# perform a series of dilations and erosions to remove any small
	# blobs left in the mask
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
			        cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]

	# only proceed if at least one contour was found
        if len(cnts) > 0:
	    # find the largest contour in the mask, then use it to compute
	    # the minimum enclosing circle and centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            (cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

	    # only draw the enclosing circle and text if the radious meets
	    # a minimum size
            if radius > 10:
                colorText = colorName + ": " + str(x)
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.putText(frame, colorText, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
			    1.0, (0, 255, 255), 2)

                if doorOriginalPos is None:
                    doorOriginalPos = (x, y)

                if x - doorOriginalPos[0] > threshold and alarmStartTime is None:
                    alarmStartTime = datetime.datetime.now()
                    os.system("mpg321 --stereo alarm.mp3")

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
vs.release()
cv2.destroyAllWindows()
