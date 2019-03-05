from random import randrange, random
import numpy as np
import matplotlib.pyplot as plt


def bbox_iou(bboxA, bboxB):
    # compute the intersection over union of two bboxes
    # I've adapted this code from the M1 base code. The function expects [tly, tlx, width, height],
    # where tl indicates the top left corner of the box.
    bboxA[2] = bboxA[0] + bboxA[2]
    bboxA[3] = bboxA[1] + bboxA[3]
    bboxB[2] = bboxB[0] + bboxB[2]
    bboxB[3] = bboxB[1] + bboxB[3]

    xA = max(bboxA[1], bboxB[1])
    yA = max(bboxA[0], bboxB[0])
    xB = min(bboxA[3], bboxB[3])
    yB = min(bboxA[2], bboxB[2])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

    try:
        iou = interArea / float(bboxAArea + bboxBArea - interArea)
    except ZeroDivisionError:
        iou = 0.0

    # return the intersection over union value
    return iou


def get_gt_bboxes(discard_probability=0.1, noise_range=35):
    """
    Creates a dictionary with the bounding boxes in each frame where the frame number is the key. It also creates
    the noisy bounding boxes.
    :return:
    """
    with open('../datasets/AICity_data/train/S03/c010/gt/gt.txt') as f:
        lines = f.readlines()
        bboxes = dict()
        bboxes_noisy = dict()
        num_of_instances = 0
        for line in lines:
            num_of_instances += 1
            line = (line.split(','))
            if line[0] in bboxes.keys():
                bboxes[line[0]].append([int(elem) for elem in line[1:6]])
            else:
                bboxes[line[0]] = [[int(elem) for elem in line[1:6]]]
            if random() > discard_probability:
                if line[0] in bboxes_noisy.keys():
                    bboxes_noisy[line[0]].append([int(elem) + randrange(-noise_range, noise_range) for elem in line[1:6]])
                else:
                    bboxes_noisy[line[0]] = [[int(elem) + randrange(-noise_range, noise_range) for elem in line[1:6]]]

    return bboxes, bboxes_noisy, num_of_instances


def evaluation_detections(thresholds, bboxes_gt, bboxes_detected, num_instances):
    """
    Computes all the scores.
    """
    TP = np.zeros(len(thresholds), dtype=int)
    FP = np.zeros(len(thresholds), dtype=int)

    scores_detections = [[] for i in range(len(thresholds))]
    # scores_detections is pair of values [result, confidence) where result is true if the example is correctly
    # classified and confidence is the confidence of the prediction. It's used to compute the precision-recall
    # curve. Not sure what it's supposed to be since we are not provided with this confidence so it's random.

    for key in bboxes_detected.keys():
        for bbox_noisy in bboxes_detected[key]:
            if key in bboxes_gt:  # if we have detected stuff and it is in the gt
                scores = [bbox_iou(bbox_noisy[1:], bbox[1:]) for bbox in bboxes_gt[key]]
                max_score = max(scores)
                for i, threshold in enumerate(thresholds):
                    if max_score > threshold:
                        TP[i] += 1
                        # we give correct boxes a slightly higher confidence score
                        scores_detections[i].append([1, random()+random()*0.1])
                    else:
                        FP[i] += 1
                        scores_detections[i].append([0, random()])
            else:  # if we have detected stuff and it is not in the gt
                for i, threshold in enumerate(thresholds):
                    FP[i] += 1
                    scores_detections[i].append([0, random()])

    FN = abs(num_instances - (TP + FP))  # number of instances not detected
    return TP, FP, FN, np.array(scores_detections)


def plot_pr(pr_over_time, thresholds, pinterps, idxs_interpolations, APs):
    for i, pr in enumerate(pr_over_time):
        plt.plot(pr[:, 1], pr[:, 0], label="PR curve at threshold: %.2f" % thresholds[i] + " (AP:%.2f)" % APs[i])
        recall = pr[idxs_interpolations[i], 1]
        plt.scatter(recall, pinterps[i], c='r')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves")
    plt.legend(loc='best')
    plt.show()


def compute_map(scores, num_instances):
    """
    Computes mAP and returns a bunch of stuff to plot the interpolated precisions
    """
    pr = []
    for i, score in enumerate(scores):
        score = score[np.argsort(-score[:, 1])]
        FP = 0
        TP = 0
        pr_ = []
        for prediction in score:
            if prediction[0]:
                TP += 1
            else:
                FP += 1
            pr_.append([TP/(TP+FP), (TP/num_instances)])
        pr.append(pr_)
    pr = np.array(pr)

    pinterps = []  # list of interpolated precisions for every threshold
    ranks = np.linspace(0, 1, 11)
    idxs_interpolations = []  # list of indexes of the interpolated precisions, just to plot the recall
    for pr_ in pr:
        pinterp = []
        idxs = []
        last_idx = -1
        for rank in ranks:
            idx = (np.abs(pr_[:, 1] - rank)).argmin()  # find the closest recall to the rank

            if rank > pr_[idx, 1]:  # this makes sure we are taking the closest recall at the right of the rank
                if idx+1 < pr_[:, 0].shape[0]:
                    idx += 1
            interpolated_precision = np.max(pr_[idx:, 0])  # find the max precision within the interval
            if idx == last_idx:  # just some checks for when the recall doesn't exist
                pinterp[-1] = 0
                idxs[-1] = 0
                pinterp.append(0)
                idxs.append(0)
            else:
                pinterp.append(interpolated_precision)
                idxs.append(idx)
            last_idx = idx
        pinterps.append(pinterp)
        idxs_interpolations.append(idxs)
    APs = np.array(pinterps).mean(axis=1)  # the AP is the average of the interpolated precisions
    mAP = APs.mean()                       # mAP is the mean of all the APs

    return pr, pinterps, idxs_interpolations, mAP, APs

