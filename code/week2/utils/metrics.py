import numpy as np


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


def evaluation_detections(thresholds, bboxes_gt, bboxes_detected, num_instances_gt, starting_frame):
    """
    Computes all the scores. Starting frame points out the frame where we start computing the scores (should be
    the start of the validation split).
    The bboxes from the detections must have this format:
    [tlx, tly, brx, bry, confidence]
    The bboxes from the gt have this format:
    [id, tlx, tly, brx, bry, confidence]
    """
    TP = np.zeros(len(thresholds), dtype=int)
    FP = np.zeros(len(thresholds), dtype=int)

    scores_detections = [[] for i in range(len(thresholds))]
    for key in bboxes_detected.keys():
        if int(key) > starting_frame:
            for bbox_detected in bboxes_detected[key]:
                if key in bboxes_gt:  # if we have detected stuff in this frame and it is in the gt
                    scores = [[bbox_iou(bbox_detected[0:4], bbox_gt[1:5]), bbox_detected[4]] for bbox_gt in bboxes_gt[key]]
                    scores_arr = np.array(scores)
                    index = scores_arr[:, 0].argmax()  # find the max iou score
                    max_score = scores[index]  # [iou value, confidence score

                    for i, threshold in enumerate(thresholds):
                        if max_score[0] > threshold:
                            TP[i] += 1
                            scores_detections[i].append([1, max_score[1]])

                            # delete the bounding box from the gt
                            del (bboxes_gt[key][index])
                            if bboxes_gt[key] == []:
                                bboxes_gt.pop(key, None)
                        else:
                            FP[i] += 1
                            scores_detections[i].append([0, max_score[1]])
                else:  # if we have detected stuff and it is not in the gt
                    for i, threshold in enumerate(thresholds):
                        FP[i] += 1
                        scores_detections[i].append([0, bbox_detected[4]])

    FN = num_instances_gt - TP  # number of instances not detected

    return TP, FP, FN, np.array(scores_detections)


def compute_map(scores, num_instances):
    """
    Computes mAP and returns a bunch of stuff to plot the interpolated precisions
    """
    pr = []
    for i, score in enumerate(scores):
        score = score[np.argsort(-score[:, 1])]  # sort by confidence score
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

    pinterps = []  # lists of interpolated precisions for every confidence level
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

