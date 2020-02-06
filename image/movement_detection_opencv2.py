#!/usr/bin/env python
import cv2
import imutils

def save_webcam(mirror=False):
    cap = cv2.VideoCapture(0) # Capturing video from webcam:
    currentFrame = 0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # Get current width of frame
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # Get current height of frame
    old = None

    while (cap.isOpened()): # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if mirror == True: 
                # Mirror the output video frame
                frame = cv2.flip(frame, 1)
            # Saves for video
            readed_frame = frame.copy()
            if old is not None:
                if old.shape[:2] == frame.shape[:2]:
                    mask = cv2.absdiff(frame, old)
                    #mask = cv2.erode(mask, None, iterations=1)
                    cv2.imshow("movement", mask)
                    mask = cv2.threshold(mask.copy(), 5, 255, cv2.THRESH_BINARY)[1]
                    cv2.imshow("tresh", mask)
                    mask = cv2.erode(mask, None, iterations=2)
                    cv2.imshow("erode", mask)
                    mask = cv2.dilate(mask, None, iterations=7)
                    cv2.imshow("dilate", mask)
                    mask = cv2.medianBlur(mask, 17, 0)
                    mask = cv2.cvtColor(mask.copy(), cv2.COLOR_BGR2GRAY)
                    cv2.imshow("computed_movement", mask)
                    #mask = cv2.erode(tresh_frame2gray, None, iterations=1)
                    #mask = cv2.dilate(mask, None, iterations=2)
                    #mask = cv2.medianBlur(mask, 3, 0)
                    # new_old = cv2.equalizeHist(old.copy())
                    # new_gray = cv2.equalizeHist(gray.copy())
                    # cv2.imshow('equalized gray image', new_gray)
                    # new_diff = cv2.absdiff(new_gray, new_old)
                    # cv2.imshow('new_diff', new_diff)
                    # diff = cv2.absdiff(old, gray)
                    # cv2.imshow('abs_diff', diff)
                    # new_tresh = cv2.threshold(new_diff.copy(), 20, 255, cv2.THRESH_BINARY)[1]
                    # new_tresh = cv2.medianBlur(new_tresh, 13, 0)
                    # if diff.mean() > 0.2:
                    #     print("Motion detected")
                    # else:
                    #     print("Nothing is moving")
                    # cv2.imshow('diff', new_tresh)

                    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = imutils.grab_contours(cnts)
                    for c in cnts:
                        if cv2.contourArea(c) < 1000:
                            continue
                        (x, y, w, h) = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('frame', frame)
            if currentFrame:
                old = readed_frame
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'): # if 'q' is pressed thenquit
            break
        # To stop duplicate images
        currentFrame += 1
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def main():
    save_webcam(mirror=True)

if __name__ == '__main__':
    main()
