Source: https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

import cv2
import sys

if __name__ == '__main__' :
    tracker = cv2.Tracker_create("MIL")
    video = cv2.VideoCapture("/Volumes/16 DOS/Python/JimCarrey.mp4")
    ok, frame = video.read()
    bbox = (360, 320, 230, 230)
    ok = tracker.init(frame, bbox)
    while True:
        ok, frame = video.read()
        if not ok:
            break
        ok, bbox = tracker.update(frame)
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0),2)
        cv2.imshow("Tracking", frame)
        k = cv2.waitKey(1) & 0xff
        if k == 100 : break
