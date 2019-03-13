import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np

from utils.gt import get_gt_bboxes
from utils.metrics import evaluation_detections, compute_map
from utils.plot import plot_pr, show_bboxes

fgbg_Mog  = cv2.bgsegm.createBackgroundSubtractorMOG(history = 200,nmixtures = 6)
fgbg_Mog2 = cv2.createBackgroundSubtractorMOG2(history = 200 )

def bgEstimate(path, alpha=2, mask_roi=None, algo = 'Mog'):
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

    # Extract foreground from remaining frames
    # plt.imshow(cv2.resize(varianceImage, (0,0), fx=rescaling_factor, fy=rescaling_factor))
    capture.set(cv2.CAP_PROP_POS_FRAMES, trainingFrames)
    detections = dict()
    nF = 0
    while success:
        nF+=1
        success, srcFrame = capture.read()
        if not success:
            break
        currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        print('frame: ', currentFrame)
        
        frame = srcFrame
        frame = cv2.GaussianBlur(frame,(19,19),0)
        
        if algo == 'Mog':
            foregroundMask = fgbg_Mog.apply(frame)
        elif algo == 'Mog2':
            foregroundMask = fgbg_Mog2.apply(frame, learningRate=0.01 )

        foregroundMask = cv2.bitwise_and(foregroundMask, mask_roi)
        #============================================================================
        # cv2.imshow("window", foregroundMask)
        # cv2.waitKey()
        kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        #maskResized = cv2.resize(foregroundMask,(0,0), fx=0.4, fy=0.4)
        # maskResized = foregroundMask
        # cv2.imshow("window", maskResized)
        # cv2.waitKey()
        openedMask = foregroundMask
        openedMask = cv2.dilate(openedMask,kernel,iterations = 2)
        for cl in range(2):
            openedMask = cv2.morphologyEx(openedMask, cv2.MORPH_CLOSE, kernel)

        _, openedMask = cv2.threshold(openedMask, 127, 255, cv2.THRESH_BINARY)
        # cv2.imshow("window", openedMask)
        # cv2.waitKey()
        # openedMask = foregroundMask
        openedMask = np.uint8(openedMask)

        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(openedMask, 4, cv2.CV_32S)

        min_size = 1000
        max_size = 1000000
        boxes    = []

        for i in range(nlabels):
            if stats[i][4] >= min_size and stats[i][4] <= max_size and (stats[i, cv2.CC_STAT_WIDTH] / stats[i, cv2.CC_STAT_HEIGHT]) < 2.8 :
                bbox = [stats[i, cv2.CC_STAT_LEFT],
                                    stats[i, cv2.CC_STAT_TOP],
                                    stats[i, cv2.CC_STAT_WIDTH],
                                    stats[i, cv2.CC_STAT_HEIGHT]]
                boxes.append(bbox)

        is_valid = [True] * len(boxes)

        for i, a in enumerate(boxes):
            for j, b in enumerate(boxes):

                if j>i:
                    a_area = a[2]*a[3]
                    b_area = b[2]*b[3]
                    aa = a.copy()
                    bb = b.copy()
                    aa[2] = a[0] + a[2]
                    aa[3] = a[1] + a[3]
                    bb[2] = b[0] + b[2]
                    bb[3] = b[1] + b[3]                
                    xA = max(aa[1], bb[1])
                    yA = max(aa[0], bb[0])
                    xB = min(aa[3], bb[3])
                    yB = min(aa[2], bb[2])
                    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                    print(a_area,b_area,interArea)
                    #print
                    if (interArea / a_area) > 0.3 or (interArea / b_area) > 0.3:
                        #print('==========================================')
                        if a_area > b_area:
                            is_valid[j] = False
                        else:
                            is_valid[i] = False
        print(is_valid)

        for i, bbox in enumerate(boxes):
            if is_valid[i]:
                cv2.rectangle(srcFrame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0,0,255), 7)

                # add the detections in the dictionary
                content = bbox
                content.extend([1.])  # confidence
                if str(currentFrame) in detections.keys():
                    detections[str(currentFrame)].append(content)
                else:
                    detections[str(currentFrame)] = [content]
        cv2.imshow('output',srcFrame)
        cv2.imshow('for',openedMask)
        cv2.waitKey(1)
        #if nF == 100:
        #    break


    capture.release()
    return detections, trainingFrames


def main():
    path = '../../datasets/AICity_data/train/S03/c010/vdo.avi'
    path_roi = '../../datasets/AICity_data/train/S03/c010/roi.jpg'
    mask_roi = cv2.imread(path_roi, cv2.IMREAD_GRAYSCALE)
    bboxes_detected, trainingFrames = bgEstimate(path, alpha=2, mask_roi=mask_roi, algo = 'Mog2')

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

