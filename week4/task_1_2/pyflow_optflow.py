from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
from pyflow import pyflow
import cv2
import sys
sys.path.insert(0,'../')
from metrics import flow_metrics

parser = argparse.ArgumentParser(
    description='Demo for python wrapper of Coarse2Fine Optical Flow')
parser.add_argument(
    '-viz', dest='viz', action='store_true',
    help='Visualize (i.e. save) output of flow.')
parser.add_argument('-alg', type=str, default='pyflow', help='')

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


def coarse2Fine(prev_frame, curr_frame, args):
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    prev_frame_gray = prev_frame_gray[:, :, np.newaxis]
    curr_frame_gray = curr_frame_gray[:, :, np.newaxis]

    prev_array = np.asarray(prev_frame_gray)
    curr_array = np.asarray(curr_frame_gray)

    # prev_array = np.asarray(prev_frame)
    # curr_array = np.asarray(curr_frame)

    prev_array = prev_array.astype(float) / 255.
    curr_array = curr_array.astype(float) / 255.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        prev_array, curr_array, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    e = time.time()
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, prev_array.shape[0], prev_array.shape[1], prev_array.shape[2]))
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)

    array_ones = np.ones((u.shape[0], u.shape[1]))
    flow_return = np.ndarray((u.shape[0], u.shape[1], 3))
    flow_return[:,:,0] = u
    flow_return[:,:,1] = v
    flow_return[:,:,2] = array_ones
    # np.save('examples/outFlow.npy', flow)

    if args.viz:
        hsv = np.zeros(prev_frame.shape, dtype=np.uint8)
        hsv[:, :, 0] = 255
        hsv[:, :, 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite('output/pyflow.png', rgb)
        # cv2.imwrite('examples/car2Warped_new.jpg', im2W[:, :, ::-1] * 255)
        cv2.imshow("optical flow", rgb)
        cv2.waitKey()

    return flow, flow_return


def farneback(prev_frame, curr_frame, args):
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    s = time.time()
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    e = time.time()
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, prev_gray.shape[0], prev_gray.shape[1], 1))
    array_ones = np.ones((flow.shape[0], flow.shape[1]))
    flow_return = np.ndarray((flow.shape[0], flow.shape[1], 3))
    flow_return[:,:,0] = flow[:,:,0]
    flow_return[:,:,1] = flow[:,:,1]
    flow_return[:,:,2] = array_ones

    if args.viz:
        u = flow[:,:,0]
        v = flow[:,:,1]
        print_flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        hsv = np.zeros(prev_frame.shape, dtype=np.uint8)
        hsv[:, :, 0] = 255
        hsv[:, :, 1] = 255
        mag, ang = cv2.cartToPolar(print_flow[..., 0], print_flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite('output/farneback.png', rgb)
        # cv2.imwrite('examples/car2Warped_new.jpg', im2W[:, :, ::-1] * 255)
        cv2.imshow("optical flow", rgb)
        cv2.waitKey()
        # cv2.imshow('Farneback', draw_flow(curr_gray, flow))
        # cv2.waitKey()

    return flow, flow_return


def lk_algorithm(prev_frame, curr_frame, args):
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    mask = np.zeros_like(prev_gray)


    # motion = np.zeros([prev_gray.shape[0], prev_gray.shape[1], 3])
    # for x in range(motion.shape[0]):
    #     for y in range(motion.shape[1]):
    #         check = (x,y)
    #         for idx, good_point in enumerate(p0):
    #             # print(int(good_point[0][0]))
    #             # print(int(good_point[0][0]))
    #             # print(check[0])
    #             # print(check[1])
    #             if int(good_point[0][0]) == check[0] and int(good_point[0][1]) == check[1]:
    #                 motion[x, y] = np.array([good_point[0][0]-p1[idx][0][0], good_point[0][1]-p1[idx][0][1], 1])
    #                 break
    #             else:
    #                 motion[x, y] = np.array([0,0,0])
    # print("hola")





    array_ones = np.ones((flow.shape[0], flow.shape[1]))
    flow_return = np.ndarray((flow.shape[0], flow.shape[1], 3))
    flow_return[:,:,0] = flow[:,:,0]
    flow_return[:,:,1] = flow[:,:,1]
    flow_return[:,:,2] = array_ones

    if args.viz:
        cv2.imshow('LK', draw_flow(curr_gray, flow))
        cv2.waitKey()

    return flow, flow_return


def draw_flow(img, flow, step=8):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def switch_alg(argument):
    switcher = {
        'pyflow': coarse2Fine,
        'farneback': farneback,
        'lk': lk_algorithm,
    }
    func = switcher[argument]
    return func


if __name__ == '__main__':

    args = parser.parse_args()

    video_dir = '../../datasets/AICity_data/train/S03/c010/vdo.avi'
    capture = cv2.VideoCapture(video_dir)
    frame_idx = 0

    opt_flow_alg = switch_alg(args.alg)

    frame1_rgb = cv2.imread('../000045_10.png')
    frame2_rgb = cv2.imread('../000045_11.png')
    gt = read_file('../gt_000045_10.png')

    # frame1_rgb = cv2.imread('../000157_10.png')
    # frame2_rgb = cv2.imread('../000157_11.png')
    # gt = read_file('../gt_000157_10.png')


    _,flow = opt_flow_alg(frame1_rgb, frame2_rgb, args)
    msen, pepn, img_err, vect_err = flow_metrics(flow, gt)
    print("Results with "+args.alg+" algorithm--> MSEN:", msen, "PEPN:",pepn)


    #### Whole Video Sequence
    # resize_factor = 0.35
    #
    # current_frame = None
    # previous_frame = None
    # while True:
    #     success, frame = capture.read()
    #     if not success:
    #         break
    #     frame_idx += 1
    #     if frame_idx == 1:
    #         previous_frame = frame
    #         previous_frame = cv2.resize(previous_frame, (0, 0), fx=resize_factor, fy=resize_factor)
    #         continue
    #     else:
    #         current_frame = frame
    #         current_frame = cv2.resize(current_frame, (0, 0), fx=resize_factor, fy=resize_factor)
    #
    #     opt_flow_alg(previous_frame, current_frame, args)
    #####