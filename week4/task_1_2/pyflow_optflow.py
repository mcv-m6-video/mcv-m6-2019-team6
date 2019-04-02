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

parser = argparse.ArgumentParser(
    description='Demo for python wrapper of Coarse2Fine Optical Flow')
parser.add_argument(
    '-viz', dest='viz', action='store_true',
    help='Visualize (i.e. save) output of flow.')
parser.add_argument('-alg', type=str, default='pyflow', help='')

def coarse2Fine(prev_frame, curr_frame, args):

    prev_array = np.asarray(prev_frame)
    curr_array = np.asarray(curr_frame)

    prev_array = prev_array.astype(float) / 255.
    curr_array = curr_array.astype(float) / 255.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        prev_array, curr_array, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    e = time.time()
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, prev_array.shape[0], prev_array.shape[1], prev_array.shape[2]))
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    # np.save('examples/outFlow.npy', flow)

    if args.viz:
        hsv = np.zeros(prev_array.shape, dtype=np.uint8)
        hsv[:, :, 0] = 255
        hsv[:, :, 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # cv2.imwrite('examples/outFlow_new.png', rgb)
        # cv2.imwrite('examples/car2Warped_new.jpg', im2W[:, :, ::-1] * 255)
        cv2.imshow("optical flow", rgb)
        cv2.waitKey()

    previous_frame = current_frame

def farneback(prev_frame, curr_frame, args):
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    if args.viz:
        cv2.imshow('Farneback', draw_flow(curr_gray, flow))
        cv2.waitKey()

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
        'LK': "LK",
    }
    func = switcher[argument]
    return func


if __name__ == '__main__':

    args = parser.parse_args()

    video_dir = '../../datasets/AICity_data/train/S03/c010/vdo.avi'
    capture = cv2.VideoCapture(video_dir)
    frame_idx = 0

    opt_flow_alg = switch_alg(args.alg)
    resize_factor = 0.35

    current_frame = None
    previous_frame = None
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_idx += 1
        if frame_idx == 1:
            previous_frame = frame
            previous_frame = cv2.resize(previous_frame, (0, 0), fx=resize_factor, fy=resize_factor)
            continue
        else:
            current_frame = frame
            current_frame = cv2.resize(current_frame, (0, 0), fx=resize_factor, fy=resize_factor)

        opt_flow_alg(previous_frame, current_frame, args)