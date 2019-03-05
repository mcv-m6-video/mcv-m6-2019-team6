import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np
import collections

from utils import bbox_iou, get_gt_bboxes, get_gt_bboxes_task2


def show_bboxes(path, bboxes, bboxes_noisy):
    """
    shows the ground truth and the noisy bounding boxes
    :param path:
    :param bboxes:
    :param bboxes_noisy:
    :return:
    """
    capture = cv2.VideoCapture(path)
    print("number of frames :", capture.get(cv2.CAP_PROP_FRAME_COUNT))
    success = True
    plt.axes()
    while success:
        success, frame = capture.read()
        current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        print('frame: ', current_frame)
        if current_frame < 218:  # skip the firsts frames without bboxes
            continue
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if str(current_frame) in bboxes.keys():
            for bbox in bboxes[str(current_frame)]:
                rect = pat.Rectangle((bbox[1], bbox[2]), bbox[3], bbox[4],
                                     linewidth=1, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
        if str(current_frame) in bboxes_noisy.keys():
            for bbox_noisy in bboxes_noisy[str(current_frame)]:
                rect = pat.Rectangle((bbox_noisy[1], bbox_noisy[2]), bbox_noisy[3], bbox_noisy[4],
                                     linewidth=1, edgecolor='b', facecolor='none')
                plt.gca().add_patch(rect)
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()

    capture.release()


def main():
    bboxes, bboxes_noisy, num_instances, dict_of_instances = get_gt_bboxes_task2(discard_probability=0.5, noise_range=25)
    path = '../datasets/AICity_data/train/S03/c010/vdo.avi'

    show = False
    if show:
        show_bboxes(path, bboxes, bboxes_noisy)

    thresholds = np.linspace(0.5, 1, 11)
    # thresholds = np.linspace(0.5, 0.5, 1)
    TP = dict()
    FP = dict()


    for key in bboxes_noisy.keys():
        tp_ind = np.zeros(len(thresholds), dtype=int)
        fp_ind = np.zeros(len(thresholds), dtype=int)
        for bbox_noisy in bboxes_noisy[key]:
            scores = [bbox_iou(bbox_noisy[1:], bbox[1:]) for bbox in bboxes[key]]
            max_score = max(scores)
            for i, threshold in enumerate(thresholds):
                if max_score > threshold:
                    tp_ind[i] += 1
                else:
                    fp_ind[i] += 1
        TP[key] = tp_ind
        FP[key] = fp_ind

    precision_measure = dict()
    recall_measure = dict()
    f1_measure = dict()

    for key in TP.keys():
        prec = np.zeros(len(thresholds), dtype=float)
        recall = np.zeros(len(thresholds), dtype=float)
        f1 = np.zeros(len(thresholds), dtype=float)

        for i in range(len(TP[key])):
            t_p = float(TP[key][i])
            f_p = float(FP[key][i])
            if t_p+f_p == 0.0:
                prec[i] = 0.0
            else:
                prec[i] = float(t_p)/float(t_p+f_p)

            f_n = dict_of_instances[key]-(TP[key][i]+FP[key][i])

            if float(t_p + f_n) == 0.0:
                recall[i] = 0.0
            else:
                recall[i] = float(t_p) / float(t_p + f_n)

            if prec[i]+recall[i] == 0.0:
                f1[i] = 0.0
            else:
                f1[i] = (2*(prec[i]*recall[i]))/(prec[i]+recall[i])

        precision_measure[key] = prec
        recall_measure[key] = recall
        f1_measure[key] = f1

    largest_index = max(list(map(int,bboxes.keys())))
    idxs_toComplete = np.arange(0, largest_index+1, 1)

    for index in idxs_toComplete:
        add_zeros = np.zeros(len(thresholds), dtype=float)
        index = str(index)
        if index not in precision_measure.keys():
            precision_measure[index] = add_zeros
        if index not in recall_measure.keys():
            recall_measure[index] = add_zeros
        if index not in f1_measure.keys():
            f1_measure[index] = add_zeros


    sorted_f1 = []
    for key, value in f1_measure.items():
        temp = [int(key), value]
        sorted_f1.append(temp)
    sorted_f1.sort( key=lambda l:l[0])

    sorted_recall = []
    for key, value in recall_measure.items():
        temp = [int(key), value]
        sorted_recall.append(temp)
    sorted_recall.sort( key=lambda l:l[0])

    sorted_precision = []
    for key, value in precision_measure.items():
        temp = [int(key), value]
        sorted_precision.append(temp)
    sorted_precision.sort(key=lambda l:l[0])

    # t_axis = np.arange(0, largest_index+1, 1)
    t_axis = np.arange(200, 401, 1)
    #CHECK NUMBER OF FRAMES TO SHOW
    if len(t_axis)<len(sorted_f1):
        sorted_f1 = sorted_f1[t_axis[0]:t_axis[-1]+1]
        sorted_precision = sorted_precision[t_axis[0]:t_axis[-1]+1]
        sorted_recall = sorted_recall[t_axis[0]:t_axis[-1]+1]

    toPlot = []
    for idx in range(len(thresholds)):
        toPlot.clear()
        for whole_element in sorted_f1:
            # print(whole_element[1][idx])
            toPlot.append(whole_element[1][idx])
        plt.plot(t_axis, toPlot)


    # plt.plot(t_axis, sorted_f1[1])


    # FN = num_instances - (TP+FP)  # number of instances not detected
    # for i, threshold in enumerate(thresholds):
    #     print("Threshold: %.2f:" % threshold, " TP:", str(TP[i]), " FP: ", str(FP[i]), " FN: ",
    #           str(FN[i]), " (", str(FP[i]+TP[i]+FN[i]), " anotations)")



if __name__ == '__main__':
    main()

