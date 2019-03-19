import copy

from random import randrange, random
import numpy as np
from xml.dom import minidom
import os

def bbox_iou(src_bboxA, src_bboxB):
    # compute the intersection over union of two bboxes
    # I've adapted this code from the M1 base code. The function expects [tly, tlx, width, height],
    # where tl indicates the top left corner of the box.
    bboxA = copy.deepcopy(src_bboxA)
    bboxB = copy.deepcopy(src_bboxB)

    id = 0
    if len(bboxB) == 2:
        id = bboxB[0]
        bboxB = bboxB[1]


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
    return iou, id

def read_xml_annotations(annotations_path):
    files = os.listdir(annotations_path)

    bboxes = dict()
    for file_ in files:
        xmldoc = minidom.parse(annotations_path + file_)
        bboxes_list = xmldoc.getElementsByTagName('box')
        for element in bboxes_list:
            frame = element.getAttribute('frame')
            xtl = int(float(element.getAttribute('xtl')))
            ytl = int(float(element.getAttribute('ytl')))
            xbr = int(float(element.getAttribute('xbr')))
            ybr = int(float(element.getAttribute('ybr')))
            width = ybr - ytl
            height = xbr - xtl

            if frame in bboxes.keys():
                bboxes[frame].append(['car', xtl, ytl, height, width, random()])
            else:
                bboxes[frame] = [['car', xtl, ytl, height, width, random()]]
    return bboxes