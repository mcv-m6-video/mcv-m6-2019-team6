import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from utils import bbox_iou, get_gt_bboxes


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
        if current_frame < 218:  # skip the first frames without bboxes
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
    bboxes, bboxes_noisy = get_gt_bboxes()
    path = '../datasets/AICity_data/train/S03/c010/vdo.avi'

    show = False
    if show:
        show_bboxes(path, bboxes, bboxes_noisy)

    TP = 0
    FP = 0
    FN = 0
    for key in bboxes.keys():
        for bbox in bboxes[key]:
            if key not in bboxes_noisy.keys():
                FN += 1
            else:
                scores = [bbox_iou(bbox[1:], bbox_noisy[1:]) for bbox_noisy in bboxes_noisy[key]]
                max_score = max(scores)
                if max_score > 0.5:
                    TP += 1
                else:
                    FP += 1
    print("TP:", str(TP), " FP: ", str(FP), " FN: ", str(FN))

if __name__ == '__main__':
    main()

