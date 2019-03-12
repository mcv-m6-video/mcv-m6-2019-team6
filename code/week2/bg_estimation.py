import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np

from utils.gt import get_gt_bboxes
from utils.metrics import evaluation_detections, compute_map
from utils.plot import plot_pr, show_bboxes

def bgEstimate(path):
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
    sequence = []
    meanImage = np.zeros((height, width), dtype='float')
    varianceImage = np.zeros((height, width), dtype='float')
    currentFrame = 0
    while success and currentFrame < trainingFrames:
        success, frame = capture.read()
        currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sequence.append(grayFrame)
        print('frame: ', currentFrame)

    # Compute mean and variance
    print('Computing mean and variance...')
    meanImage = np.mean(sequence, axis=0)
    varianceImage = np.var(sequence, axis=0)

    # Extract foreground from remaining frames
    alpha = 7
    print(meanImage.shape)
    # plt.imshow(cv2.resize(varianceImage, (0,0), fx=rescaling_factor, fy=rescaling_factor))
    detections = dict()
    while success:
        success, frame = capture.read()
        if not success:
            break
        currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print('frame: ', currentFrame)
        probabilityImage = np.absolute(grayFrame - meanImage) - alpha * (np.sqrt(varianceImage)+ 2)
        retVal, foregroundMask = cv2.threshold(probabilityImage, 0, 255, cv2.THRESH_BINARY)
        # cv2.imshow("window", foregroundMask)
        # cv2.waitKey()        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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

        min_thresh = 2000
        max_thresh = 1000000
        for i in range(nlabels):
            if stats[i][4] >= min_thresh and stats[i][4] <= max_thresh:
                bbox = [stats[i, cv2.CC_STAT_LEFT],
                                    stats[i, cv2.CC_STAT_TOP], 
                                    stats[i, cv2.CC_STAT_WIDTH], 
                                    stats[i, cv2.CC_STAT_HEIGHT]]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0,0,255), 7)

                # add the detections in the dictionary
                content = bbox
                content.extend([1.])  # confidence
                if str(currentFrame) in detections.keys():
                    detections[str(currentFrame)].append(content)
                else:
                    detections[str(currentFrame)] = [content]

        # plt.imshow(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (0,0), fx=rescaling_factor, fy=rescaling_factor))
        # plt.show(block=False)
        # plt.pause(0.05)
        # plt.clf()

    # plt.imshow(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (0,0), fx=rescaling_factor, fy=rescaling_factor))
    
    capture.release()
    return detections, trainingFrames


def main():

    path = '../../datasets/AICity_data/train/S03/c010/vdo.avi'
    bboxes_detected, trainingFrames = bgEstimate(path)

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

