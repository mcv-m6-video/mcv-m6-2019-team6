import matplotlib.pyplot as plt
import matplotlib.patches as pat
import cv2

from .metrics import bbox_iou


def plot_pr(pr_over_time, thresholds, pinterps, idxs_interpolations, APs):
    for i, pr in enumerate(pr_over_time):
        plt.plot(pr[:, 1], pr[:, 0], label="P-R at thres.: %.2f" % thresholds[i] + " (AP:%.2f)" % APs[i])
        recall = pr[idxs_interpolations[i], 1]
        plt.scatter(recall, pinterps[i], c='r')
    for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        plt.scatter(i, 0.0, c='r')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves")
    plt.legend(loc='best')
    plt.show()


def show_bboxes(path, bboxes, bboxes_detected):
    """
    shows the ground truth and the noisy bounding boxes
    :param path:
    :param bboxes:
    :param bboxes_detected:
    :return:
    """
    capture = cv2.VideoCapture(path)
    print("number of frames :", capture.get(cv2.CAP_PROP_FRAME_COUNT))
    success = True
    plt.axes()
    rescaling_factor = 0.4
    while success:
        success, frame = capture.read()
        if not success:
            break
        current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        print('frame: ', current_frame)
        if current_frame < 500:  # skip the firsts frames without bboxes
            continue
        plt.imshow(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (0,0), fx=rescaling_factor, fy=rescaling_factor))
        if str(current_frame) in bboxes.keys():
            for bbox in bboxes[str(current_frame)]:
                rect = pat.Rectangle((bbox[1]*rescaling_factor, bbox[2]*rescaling_factor),
                                     bbox[3]*rescaling_factor, bbox[4]*rescaling_factor,
                                     linewidth=1, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
                if str(current_frame) in bboxes_detected.keys():
                    scores = [bbox_iou(bbox[1:], bbox_noisy[0:]) for bbox_noisy in bboxes_detected[str(current_frame)]]
                    max_score = max(scores)
                else:
                    max_score = 0.0
                plt.text(bbox[1]*rescaling_factor + bbox[3]*rescaling_factor,
                         bbox[2]*rescaling_factor + bbox[4]*rescaling_factor,
                         "IoU: %.2f" % max_score, color='red', fontsize=15)
        if str(current_frame) in bboxes_detected.keys():
            for bbox_noisy in bboxes_detected[str(current_frame)]:
                rect = pat.Rectangle((bbox_noisy[0]*rescaling_factor, bbox_noisy[1]*rescaling_factor),
                                     bbox_noisy[2]*rescaling_factor, bbox_noisy[3]*rescaling_factor,
                                     linewidth=1, edgecolor='b', facecolor='none')
                plt.gca().add_patch(rect)
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()

    capture.release()

