#!pip install opencv-contrib-python
import cv2
import imutils

#initialize saliency
saliency = None

frameCnt = 1

#Grab clips
baseAddress = "testVid.mp4"
cap = cv2.VideoCapture(baseAddress)

# check that it can open
if (cap.isOpened() == False):
    print("Error opening clips file")
    exit(-1)

#read in clips
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        #resize frame
        frame = imutils.resize(frame, width=500)

        #if saliency hasn't been initialized do so
        if saliency is None:
            saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
            saliency.setImagesize(frame.shape[1], frame.shape[0])
            saliency.init()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (success, saliencyMap) = saliency.computeSaliency(gray)
        saliencyMap = (saliencyMap * 255).astype("uint8")

        frameNm = "Frame_" + str(frameCnt)
        # cv2.imshow(frameNm, frame)
        # cv2.imshow("map", saliencyMap)

        fileNm = "saliencyMaps/" + frameNm + ".jpeg"
        cv2.imwrite(fileNm, saliencyMap)
        frameCnt += 1


        #Press Q on keyboard to quit
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    else:
        break

#release the clips
cap.release()

#close all frames
cv2.destroyAllWindows()

