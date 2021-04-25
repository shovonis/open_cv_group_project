import cv2
import numpy as np


def get_optical_flow(video_path, output_file):
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    first_frame = first_frame[:, 0:256]
    prev_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255
    i_frame = 0

    # Create the output File
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    h, w, _ = first_frame.shape
    out = cv2.VideoWriter(output_file, fourcc, 20, (w, h), True)

    while cap.isOpened():
        ret, current_frame = cap.read()

        if ret:
            current_frame = current_frame[:, 0:256]
            next_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            out.write(rgb)
            prev_frame = next_frame
            i_frame += 1
        else:
            cap.release()
            break

#
# def main():
#     fileName = "file_example_AVI_480_750kB.avi"
#     get_optical_flow(fileName, output_file='output.mp4')
#
#
# if __name__ == "__main__":
#     main()
