import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np
import os


def adaptive_bg_estimation(path, alpha=2.5, rho=0.1, mask_roi=None, path_to_save=None):
    capture = cv2.VideoCapture(path)
    totalFrames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("number of frames :", totalFrames)
    print("Frame size: " + str(width) +" x " + str(height))
    success = True
    rescaling_factor = 0.4

    # Load training sequence
    trainingFrames = totalFrames*0.25
    sequence = []
    currentFrame = 0
    while success and currentFrame < trainingFrames:
        success, frame = capture.read()
        currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sequence.append(grayFrame)
        print('frame: ', currentFrame)

    print('Computing mean and variance...')
    meanImage = np.mean(sequence, axis=0)
    varianceImage = np.std(sequence, axis=0)

    # Extract foreground from remaining frames
    print(meanImage.shape)
    masks = []
    detections = dict()
    while success:
        success, frame = capture.read()
        if not success: continue
        currentFrame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print('frame: ', currentFrame)

        mask = np.absolute(grayFrame - meanImage) >= alpha * (varianceImage + 2)
        if mask_roi is not None:
            mask = mask & mask_roi

        if path_to_save is not None:
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            cv2.imwrite(os.path.join(path_to_save, str(currentFrame) + '.png'), mask.astype('uint8') * 255)

        # cv2.imshow("window", foregroundMask)
        # cv2.waitKey()

        meanImage = (1-rho)*meanImage + rho*grayFrame
        varianceImage = np.sqrt((1-rho)*np.power(varianceImage, 2) + rho*np.power((grayFrame-meanImage), 2))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        maskResized = cv2.resize(mask, (0, 0), fx=0.4, fy=0.4)
        # maskResized = foregroundMask
        # cv2.imshow("window", maskResized)
        # cv2.waitKey()
        openedMask = cv2.morphologyEx(maskResized, cv2.MORPH_CLOSE, kernel)
        openedMask = cv2.resize(openedMask, (width, height))
        # cv2.imshow("window", openedMask)
        # cv2.waitKey()
        # openedMask = foregroundMask
        openedMask = np.uint8(openedMask)*255

        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(openedMask, 4, cv2.CV_32S)

        min_thresh = 2000
        max_thresh = 1000000
        for i in range(nlabels):
            if stats[i][4] >= min_thresh and stats[i][4] <= max_thresh:
                bbox = [stats[i, cv2.CC_STAT_LEFT],
                        stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH],
                        stats[i, cv2.CC_STAT_HEIGHT]]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 7)

                # add the detections in the dictionary
                content = bbox
                content.extend([1.])  # confidence
                if str(currentFrame) in detections.keys():
                    detections[str(currentFrame)].append(content)
                else:
                    detections[str(currentFrame)] = [content]

        plt.imshow(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (0,0), fx=rescaling_factor, fy=rescaling_factor))
        plt.show(block=False)
        plt.pause(0.05)
        plt.clf()

        plt.imshow(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (0,0), fx=rescaling_factor, fy=rescaling_factor))

    capture.release()
    return detections, trainingFrames


    capture.release()
    return masks


if __name__ == "__main__":
    path = '../../datasets/AICity_data/train/S03/c010/vdo.avi'
    path_to_save_adap = '../../datasets/AICity_data/train/S03/c010/maskAdap/'
    path_roi = '../../datasets/AICity_data/train/S03/c010/roi.jpg'

    mask_roi = cv2.imread(path_roi, cv2.IMREAD_GRAYSCALE)

    adaptive_bg_estimation(path, alpha=2.5, rho=0.0, mask_roi=mask_roi, path_to_save=None)

