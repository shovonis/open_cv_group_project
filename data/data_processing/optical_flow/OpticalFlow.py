import cv2
import numpy as np
import time

def opticalFlowFunc(fileName):
    cap = cv2.VideoCapture(fileName)
    cap.set(cv2.CAP_PROP_FPS, 5)


    ret, frame1 = cap.read()
    frame1 = frame1[:, 0:256]
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    i_frame = 0  


    output_file = "./crab3_track.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    is_begin = True

    frame_rate = 20
    prev = 0
    while(1):

        time_elapsed = time.time() - prev
        ret, frame2 = cap.read()

        if time_elapsed > 1./frame_rate:
            prev = time.time()
            
        frame2 = frame2[:, 0:256]
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        if is_begin:
            h, w, _ = frame2.shape
            out = cv2.VideoWriter(output_file, fourcc, 30, (w, h), True)
            is_begin = False


        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        save_path='frame_%d' % i_frame
        #np.save(save_path, flow)
    
        
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        out.write(rgb)
        cv2.imshow('frame2',rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',rgb)
        prvs = next
        i_frame+=1


    cap.release()
    cv2.destroyAllWindows()

def main():
    fileName = "file_example_AVI_480_750kB.avi"
    opticalFlowFunc(fileName)

if __name__ == "__main__":
    main()
