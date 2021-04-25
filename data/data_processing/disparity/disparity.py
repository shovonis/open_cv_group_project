import cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_disparity_sgm(stereo_image):
    left_image = stereo_image[:, 0:256]
    right_image = stereo_image[:, 256:]

    # stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)
    #
    # disp = stereo.compute(left_image, right_image)

    # parameters for disparity
    max_disp = 32
    window_size = 3

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-max_disp,
        numDisparities=max_disp * 2,
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=1,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    lambda_value = 8000
    sigma = 1.5
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lambda_value)
    wls_filter.setSigmaColor(sigma)

    left_disp = left_matcher.compute(left_image, right_image)
    right_disp = right_matcher.compute(right_image, left_image)
    left_disp = np.int16(left_disp)
    right_disp = np.int16(right_disp)

    disparity = wls_filter.filter(left_disp, left_image, None, right_disp) / 16.0

    return disparity


### Read Image and Plot Disparity ####
# image = cv2.imread("Frame-1134-04-27-20-652.png")
#
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# disp_sgb = generate_disparity_sgm(image)
# plt.imshow(disp_sgb, 'gray')
# plt.colorbar()
# plt.show()
#
# plt.imshow(disp_bm, 'gray')
# plt.colorbar()
# plt.show()
#
