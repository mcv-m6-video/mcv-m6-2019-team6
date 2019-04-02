import cv2
import numpy as np


matching_methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                    'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


def block_matching(frame1, frame2, block_size=16, search_area=64, method='cv2.TM_CCOEFF_NORMED'):
    width = frame1.shape[1]
    height = frame1.shape[0]

    search_area = int(search_area/2)
    motion = []
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            center_i = int(round(i+(block_size/2)))  # center i of the block
            center_j = int(round(j+(block_size/2)))  # center j of the block
            block = frame1[i:i+block_size, j:j+block_size]
            # search area in i goes from center_i-search_area/2 (if > 0) to center_i+search_area/2, and the same for j
            search_space = frame2[max(0, center_i-search_area):center_i+search_area,
                                  max(0, center_j-search_area):center_j+search_area]

            meth = eval(method)
            res = cv2.matchTemplate(search_space, block, meth)
            _, _, _, max_loc = cv2.minMaxLoc(res)  # max_loc gives the upper left corner

            # TODO: the displacement at the borders is wrong because the size of the search area changes, the center of the area is different
            cent = int(search_area-(block_size/2))  # center of the search space (in its coordinates)
            displacement = np.array(max_loc) - np.array([cent, cent])  # distance from the highest response to the
                                                                       # center of the search space

            motion.append(([center_i, center_j], [displacement[1], displacement[0]]))  # for plotting purposes

    return motion


def main():

    frame1_rgb = cv2.imread('000045_10.png')
    frame2_rgb = cv2.imread('000045_11.png')
    # frame1_rgb = cv2.imread('000157_10.png')
    # frame2_rgb = cv2.imread('000157_11.png')

    frame1 = cv2.cvtColor(frame1_rgb, cv2.COLOR_RGB2GRAY)
    frame2 = cv2.cvtColor(frame2_rgb, cv2.COLOR_RGB2GRAY)

    motion = block_matching(frame1, frame2, block_size=16, search_area=64)

    for mot in motion:
        center = mot[0]
        displacement = mot[1]
        cv2.arrowedLine(frame1_rgb, (center[1], center[0]),
                        (center[1]+displacement[1]*2, center[0]+displacement[0]*2), (255, 0, 0), 1)

    cv2.imshow("motion", frame1_rgb)
    cv2.waitKey()

if __name__ == '__main__':
    main()