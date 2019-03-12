import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np

from utils.gt import get_gt_bboxes
from utils.metrics import evaluation_detections, compute_map
from utils.plot import plot_pr, show_bboxes

def bgEstimate(path, alpha=2, mask_roi=None, colorspace='GRAY'):
    """
    :param path:
    :param bboxes:
    :return:
    """
    capture = cv2.VideoCapture(path)
    totalFrames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("number of frames :", totalFrames)
    print("Frame size: " + str(width) +" x " + str(height))
    success = True
    plt.axes()
    rescaling_factor = 0.4
    counter = 0

    # Load training sequence
    trainingFrames = totalFrames * 0.25
    if colorspace == 'GRAY':
        frames_acc = np.zeros((height, width), dtype='float')
        var_acc = np.zeros((height, width), dtype='float')
    elif colorspace == 'YUV':
        frames_acc = np.zeros((height, width, 3), dtype='float')
        var_acc = np.zeros((height, width, 3), dtype='float')
    else:
        print("wrong color space")
        return

    currentFrame = 0
    while success and currentFrame < int(trainingFrames):
        success, frame = capture.read()
        currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if colorspace == 'GRAY':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif colorspace == 'YUV':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        frames_acc += frame

    # Compute mean and variance
    print('Computing mean and variance...')
    meanImage = frames_acc / trainingFrames

    # set the current frame to 0 and compute the variance
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    success = True
    while success and currentFrame < trainingFrames:
        success, frame = capture.read()
        currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if colorspace == 'GRAY':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif colorspace == 'YUV':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        var_acc += (frame - meanImage) ** 2

    varianceImage = (var_acc / trainingFrames)

    # Extract foreground from remaining frames
    # plt.imshow(cv2.resize(varianceImage, (0,0), fx=rescaling_factor, fy=rescaling_factor))
    detections = dict()
    while success:
        success, srcFrame = capture.read()
        if not success:
            break
        currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        print('frame: ', currentFrame)

        frame = srcFrame
        if colorspace == 'GRAY':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif colorspace == 'YUV':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

        probabilityImage = np.absolute(frame - meanImage) - alpha * (np.sqrt(varianceImage)+ 2)
        retVal, foregroundMask = cv2.threshold(probabilityImage, 0, 255, cv2.THRESH_BINARY)
        # cv2.imshow("window", foregroundMask)
        # cv2.waitKey()
        if colorspace != 'GRAY':
            maskYU = cv2.bitwise_and(foregroundMask[:,:,0], foregroundMask[:,:,1])
            maskYV = cv2.bitwise_and(foregroundMask[:,:,0], foregroundMask[:,:,2])
            foregroundMask = cv2.bitwise_or(maskYU[:,:], maskYV[:,:])
            if mask_roi is not None:  # does not work in my computer when I load the RoI
                foregroundMask = cv2.bitwise_and(foregroundMask[:,:], mask_roi[:,:])
        elif mask_roi is not None:
            # does not work in my computer when I load the RoI
            foregroundMask = cv2.bitwise_and(foregroundMask[:,:], mask_roi[:,:])


        # cv2.imshow("window", foregroundMask)
        # cv2.waitKey()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        maskResized = cv2.resize(foregroundMask,(0,0), fx=0.4, fy=0.4)
        # maskResized = foregroundMask
        # cv2.imshow("window", maskResized)
        # cv2.waitKey()
        openedMask = cv2.morphologyEx(maskResized, cv2.MORPH_CLOSE, kernel)
        openedMask = cv2.resize(openedMask,(width,height))
        # cv2.imshow("window", openedMask)
        # cv2.waitKey()
        # openedMask = foregroundMask
        openedMask = np.uint8(openedMask)

        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(openedMask, 4, cv2.CV_32S)

        min_size = 1000
        max_size = 1000000
        for i in range(nlabels):
            if stats[i][4] >= min_size and stats[i][4] <= max_size and (stats[i, cv2.CC_STAT_WIDTH] / stats[i, cv2.CC_STAT_HEIGHT]) < 2.8 :
                bbox = [stats[i, cv2.CC_STAT_LEFT],
                                    stats[i, cv2.CC_STAT_TOP],
                                    stats[i, cv2.CC_STAT_WIDTH],
                                    stats[i, cv2.CC_STAT_HEIGHT]]
                cv2.rectangle(srcFrame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0,0,255), 7)

                # add the detections in the dictionary
                content = bbox
                content.extend([1.])  # confidence
                if str(currentFrame) in detections.keys():
                    detections[str(currentFrame)].append(content)
                else:
                    detections[str(currentFrame)] = [content]

        plt.imshow(cv2.resize(cv2.cvtColor(srcFrame, cv2.COLOR_BGR2RGB), (0,0), fx=rescaling_factor, fy=rescaling_factor))
        plt.show(block=False)
        plt.pause(0.05)
        plt.clf()

    # plt.imshow(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (0,0), fx=rescaling_factor, fy=rescaling_factor))

    capture.release()
    return detections, trainingFrames


def main():
    path = '../../datasets/AICity_data/train/S03/c010/vdo.avi'
    path_roi = '../../datasets/AICity_data/train/S03/c010/roi.jpg'
    mask_roi = cv2.imread(path_roi, cv2.IMREAD_GRAYSCALE)
    bboxes_detected, trainingFrames = bgEstimate(path, alpha=2, mask_roi=None, colorspace='YUV')

    bboxes_gt, num_instances_gt = get_gt_bboxes()
    # get the number of instances in the validation split (needed to calculate the number of FN and the recall)
    num_instances_validation = 0
    for key in bboxes_gt.keys():
        if int(key) > trainingFrames:
            num_instances_validation += len(bboxes_gt[key])

    threshold = [0.5]
    TP, FP, FN, scores = evaluation_detections(threshold, bboxes_gt, bboxes_detected, num_instances_validation, trainingFrames)
    print("tp: ", TP, "fp: ", FP, "fn: ", FN)
    pr, pinterps, idxs_interpolations, mAP, APs = compute_map(scores, num_instances_validation)
    print("map: ", mAP)
    plot_pr(pr, threshold, pinterps, idxs_interpolations, APs)  # plot mAP

    bboxes_gt, num_instances_gt = get_gt_bboxes()  # load the bboxes from the gt again
    show_bboxes(path, bboxes_gt, bboxes_detected)  # show the bounding boxes

    return


if __name__ == '__main__':
    main()

