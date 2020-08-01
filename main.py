
import argparse
import imutils
from imutils.video import VideoStream
import cv2
import time
import datetime
import os
import threading
from flask import Response
from flask import Flask
from flask import render_template

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

mog = cv2.bgsegm.createBackgroundSubtractorMOG()
alarmStartTime = None
alarmActivatedTime = None

# define the color ranges
colorRanges = [
    ((29, 86, 6), (64, 255, 255), "door")]

vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def detect_motion():
    global vs, outputFrame, lock, alarmStartTime, alarmActivatedTime
    
    # keep looping
    while True:
        # grab the current frame
        frame = vs.read()

        # if we are viewing a video and we did not grab a frame, then we have
        # reached the end of the video
        if args.get("video") and not grabbed:
            break

        # resize the frame, blur it, and convert it to the HSV color space
        frame = imutils.resize(frame, width=600)

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()

        # convert the frame to grayscale and smoothen it using a
        # gaussian kernel
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # apply the MOG background subtraction model
        mask = mog.apply(gray)

        # apply a series of erosions to break apart connected
        # components, then find contours in the mask
        erode = cv2.erode(mask, (7, 7), iterations=2)
        cnts = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        now = datetime.datetime.now()
        if alarmStartTime is not None and (now - alarmStartTime).seconds > 5:
            alarmStartTime = None

        if alarmActivatedTime is not None and (now - alarmActivatedTime).seconds > 30:
            alarmActivatedTime = None

        # loop over each contour
        for c in cnts:
            # if the contour area is less than the minimum area
            # required then ignore the object
            area = cv2.contourArea(c)
            if area < 600:
                continue

            # compute the bounding box coordinates of the contour
            (rx, ry, rw, rh) = cv2.boundingRect(c)

            # add the bounding box coordinates to the rectangles list
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh),
                          (0, 255, 0), 2)

            debugText = str(area)
            cv2.putText(frame, debugText, (rx, rx), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 255), 2)

            if alarmActivatedTime is not None and alarmStartTime is None:
                alarmStartTime = datetime.datetime.now()
                os.system("mpg321 --stereo alarm.mp3")

        # show the frame to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock, alarmActivatedTime

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            alarmActivatedTime = datetime.datetime.now()
            
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        
            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
          bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the (optional) video file")
    ap.add_argument("-w", "--webcam", help="use webcam", action='store_true')
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion, args=())
    t.daemon = True
    t.start()
    
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

# cleanup the camera and close any open windows
vs.stop()
cv2.destroyAllWindows()
