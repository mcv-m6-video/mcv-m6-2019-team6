import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np

from utils import get_gt_bboxes, evaluation_detections, compute_map, plot_pr, read_xml_annotations


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
    rescaling_factor = 0.25
    while success:
        success, frame = capture.read()
        current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        print('frame: ', current_frame)
        if current_frame < 218:  # skip the firsts frames without bboxes
            continue
        plt.imshow(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (0,0), fx=rescaling_factor, fy=rescaling_factor))
        if str(current_frame) in bboxes.keys():
            for bbox in bboxes[str(current_frame)]:
                rect = pat.Rectangle((bbox[1]*rescaling_factor, bbox[2]*rescaling_factor),
                                     bbox[3]*rescaling_factor, bbox[4]*rescaling_factor,
                                     linewidth=1, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
        if str(current_frame) in bboxes_noisy.keys():
            for bbox_noisy in bboxes_noisy[str(current_frame)]:
                rect = pat.Rectangle((bbox_noisy[1]*rescaling_factor, bbox_noisy[2]*rescaling_factor),
                                     bbox_noisy[3]*rescaling_factor, bbox_noisy[4]*rescaling_factor,
                                     linewidth=1, edgecolor='b', facecolor='none')
                plt.gca().add_patch(rect)
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()

    capture.release()


def main():
    bboxes_gt, bboxes_noisy, num_instances_gt = get_gt_bboxes(discard_probability=0.2, noise_range=50)
    path = '../datasets/AICity_data/train/S03/c010/vdo.avi'

    show = True
    if show:
        show_bboxes(path, bboxes_gt, bboxes_noisy)

    # exercises 1.1 and 1.2
    # compute all the scores (TP, FP, FN)
    thresholds = np.linspace(0.5, 1, 11)
    TP, FP, FN, scores = evaluation_detections(thresholds, bboxes_gt, bboxes_noisy, num_instances_gt)
    for i, threshold in enumerate(thresholds):
        print("Threshold: %.2f:" % threshold, " TP:", str(TP[i]), " FP: ", str(FP[i]), " FN: ",
              str(FN[i]), " (", str(FP[i] + TP[i] + FN[i]), " annotations)")

    pr, pinterps, idxs_interpolations, mAP, APs = compute_map(scores, num_instances_gt)
    print("mAP: ", mAP)

    # plot the pr curves, the interpolated values used to compute AP appear in red:
    plot_pr(pr, thresholds, pinterps, idxs_interpolations, APs)

    # the same thing with the annotations (exercise 1.3)
    bboxes_annotations = read_xml_annotations('../annotations/')

    # compute all the scores (TP, FP, FN)
    thresholds = np.linspace(0.5, 1, 11)
    TP, FP, FN, scores = evaluation_detections(thresholds, bboxes_gt, bboxes_annotations, num_instances_gt)
    for i, threshold in enumerate(thresholds):
        print("Threshold: %.2f:" % threshold, " TP:", str(TP[i]), " FP: ", str(FP[i]), " FN: ",
              str(FN[i]), " (", str(FP[i] + TP[i] + FN[i]), " annotations)")

    pr, pinterps, idxs_interpolations, mAP, APs = compute_map(scores, num_instances_gt)
    print("mAP: ", mAP)

    # plot the pr curves, the interpolated values used to compute AP appear in red:
    plot_pr(pr, thresholds, pinterps, idxs_interpolations, APs)

    # if True:
    #     show_bboxes(path, bboxes_gt, bboxes_annotations)



if __name__ == '__main__':
    main()

