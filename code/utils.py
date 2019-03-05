from random import randrange, random


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


def get_gt_bboxes_task2(discard_probability=0.1, noise_range=35):
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
        dict_of_instances = dict() #this dict is for task2 when we need the FN frame by frame (not in total)
        for line in lines:
            num_of_instances += 1
            line = (line.split(','))
            if line[0] in bboxes.keys():
                bboxes[line[0]].append([int(elem) for elem in line[1:6]])
            else:
                bboxes[line[0]] = [[int(elem) for elem in line[1:6]]]

            if line[0] in dict_of_instances.keys():
                dict_of_instances[line[0]] += 1
            else:
                dict_of_instances[line[0]] = 1
            if random() > discard_probability:
                if line[0] in bboxes_noisy.keys():
                    bboxes_noisy[line[0]].append([int(elem) + randrange(-noise_range, noise_range) for elem in line[1:6]])
                else:
                    bboxes_noisy[line[0]] = [[int(elem) + randrange(-noise_range, noise_range) for elem in line[1:6]]]

    return bboxes, bboxes_noisy, num_of_instances, dict_of_instances

