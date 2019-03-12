

def get_gt_bboxes():
    """
    Creates a dictionary with the bounding boxes in each frame where the frame number is the key.
    :return:
    """
    with open('../../datasets/AICity_data/train/S03/c010/gt/gt.txt') as f:
        lines = f.readlines()
        bboxes = dict()
        num_of_instances = 0
        for line in lines:
            num_of_instances += 1
            line = (line.split(','))
            if line[0] in bboxes.keys():
                content = [int(elem) for elem in line[1:6]]
                bboxes[line[0]].append(content)
            else:
                content = [int(elem) for elem in line[1:6]]
                bboxes[line[0]] = [content]

    return bboxes, num_of_instances

