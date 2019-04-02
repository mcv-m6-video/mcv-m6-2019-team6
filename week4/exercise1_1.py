import cv2
import numpy as np
import matplotlib.pyplot as plt
from metrics import flow_metrics


matching_methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                    'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


def read_file(path):
    """Read an optical flow map from disk
    Optical flow maps are stored in disk as 3-channel uint16 PNG images,
    following the method described in the KITTI optical flow dataset 2012
    (http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow).
    Returns:
      numpy array with shape [height, width, 3]. The first and second channels
      denote the corresponding optical flow 2D vector (u, v). The third channel
      is a mask denoting if an optical flow 2D vector exists for that pixel.
      Vector components u and v values range [-512..512].
    """
    data = cv2.imread(path, -1).astype('float32')
    result = np.empty(data.shape, dtype='float32')
    result[:,:,0] = (data[:,:,2] - 2**15) / 64
    result[:,:,1] = (data[:,:,1] - 2**15) / 64
    result[:,:,2] = data[:,:,0]

    return result


def block_matching(frame1, frame2, block_size=16, search_area=64, method='cv2.TM_CCORR_NORMED'):
    width = frame1.shape[1]
    height = frame1.shape[0]

    search_area = int(search_area/2)
    motion_blocks = []  # for plotting purposes
    motion = np.zeros([frame1.shape[0], frame1.shape[1], 3])
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

            motion_blocks.append(([center_i, center_j], [displacement[1], displacement[0]]))
            motion[i:i+block_size, j:j+block_size] = np.array([displacement[1], displacement[0], 1])

    return motion_blocks, motion


def main():

    frame1_rgb = cv2.imread('000045_10.png')
    frame2_rgb = cv2.imread('000045_11.png')
    gt = read_file('gt_000045_10.png')
    # frame1_rgb = cv2.imread('000157_10.png')
    # frame2_rgb = cv2.imread('000157_11.png')
    # gt = read_file('gt_000157_10.png')

    frame1 = cv2.cvtColor(frame1_rgb, cv2.COLOR_RGB2GRAY)
    frame2 = cv2.cvtColor(frame2_rgb, cv2.COLOR_RGB2GRAY)

    motion_blocks, motion = block_matching(frame1, frame2, block_size=16, search_area=64)

    msen, pepn, img_err, vect_err = flow_metrics(motion.astype(np.uint8), gt)
    print("MSEN: ", msen, " - PEPN: ", pepn)

    for motion_block in motion_blocks:
        center = motion_block[0]
        displacement = motion_block[1]
        cv2.arrowedLine(frame1_rgb, (center[1], center[0]),
                        (center[1]+displacement[1]*2, center[0]+displacement[0]*2), (255, 0, 0), 1)

    cv2.imshow("motion", frame1_rgb)
    cv2.waitKey()

    plt.imshow(np.sqrt(motion[:, :, 1] ** 2 + motion[:, :, 1] ** 2))
    plt.title("module of the motion vectors")
    plt.show()

    return


if __name__ == '__main__':
    main()

