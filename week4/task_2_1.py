import cv2
import numpy as np
import matplotlib.pyplot as plt
from metrics import flow_metrics
from math import sqrt
from exercise1_1 import block_matching

def get_motion(frame1_rgb, frame2_rgb):
	# calculate motion between the two frames
    frame1 = cv2.cvtColor(frame1_rgb, cv2.COLOR_RGB2GRAY)
    frame2 = cv2.cvtColor(frame2_rgb, cv2.COLOR_RGB2GRAY)

    for block_size in [16*5]:  # search space
        for search_area in [32*5]:
            if search_area <= block_size: continue
            motion_blocks, motion = block_matching(frame1, frame2, block_size=block_size, search_area=search_area, bwd=True)
    # visualize results
    motion_module = np.sqrt(motion[:, :, 1] ** 2 + motion[:, :, 0] ** 2)
    max_motion_module = np.max(motion_module)
    min_motion_module = np.min(motion_module)

    displacements = []

    for motion_block in motion_blocks:
        center = motion_block[0]
        displacement = motion_block[1]
        displacements.append(displacement)        
        green = 255 * ((sqrt(displacement[0]**2 + displacement[1]**2) - min_motion_module) / (max_motion_module - min_motion_module))*1
        cv2.arrowedLine(frame1_rgb, (center[1], center[0]),(center[1]+displacement[1]*2,\
        center[0]+displacement[0]*2), (0, green, 255), 3,tipLength=0.4)
    
    displacements = np.array(displacements,dtype=int)
    """
    mode of the shift
    """
    displacements.sort(axis = 0)
    middle = int(displacements.shape[0]/2)
    mean_shift = [-displacements[middle,0], -displacements[middle,1]]
    
    center = (int(frame1_rgb.shape[0]/2),int(frame1_rgb.shape[1]/2))
    displacement = motion_blocks[middle][1]
    green = 255 * ((sqrt(displacement[0]**2 + displacement[1]**2) - min_motion_module) / (max_motion_module - min_motion_module))*1
    cv2.arrowedLine(frame1_rgb, (center[1], center[0]),(center[1]+displacement[1]*2,\
    center[0]+displacement[0]*2), (255, 0, 0), 12,tipLength=0.4)
    cv2.arrowedLine(frame1_rgb, (center[1], center[0]),(center[1]+displacement[1]*2,\
    center[0]+displacement[0]*2), (0, 255, 0), 6,tipLength=0.4)

    """
    mean of the shift
    """
    #mean_shift    = displacements.mean(axis=0)
    #mean_shift    = [-np.round(x) for x in mean_shift]
    #mean_shift    = [-np.round(x)*1.2 for x in mean_shift]
    #exit(0)

    return displacements, mean_shift, frame1_rgb


def shift_img(img,shift):
    num_rows, num_cols = img.shape[:2]
    translation_matrix = np.float32([ [1,0,shift[1]], [0,1,shift[0]] ])
    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
    return img_translation


def test_image():
    frame1_rgb = cv2.imread('1.png')
    frame2_rgb = cv2.imread('2.png')

    displacements, shift, motion = get_motion(frame1_rgb, frame2_rgb)
    img_translation              = shift_img( frame2_rgb, shift)

    cv2.imshow('1',frame1_rgb)
    cv2.waitKey()

    cv2.imshow('1',img_translation)
    cv2.waitKey()



def test_video():
    #video_dir = 'patio.mp4'
    video_dir = 'hippo.mp4'

    capture = cv2.VideoCapture(video_dir)
    prev_frame = None
    frame_idx  = 0 
    while True:
        success, frame = capture.read()
        frame_idx+=1
        if not success:
            break
        if prev_frame is not None:
            displacements, shift , motion = get_motion(prev_frame, frame)
            img_translation               = shift_img(frame, shift)
            prev_frame = img_translation.copy()

            cv2.rectangle(frame,(0,0),(frame.shape[1],frame.shape[0]),(255,255,0),3)
            cv2.rectangle(motion,(0,0),(frame.shape[1],frame.shape[0]),(255,255,0),3)
            cv2.rectangle(img_translation,(0,0),(frame.shape[1],frame.shape[0]),(255,255,0),3)
            #cv2.imshow('1',img_translation)
            #cv2.imshow('2',frame)
            #cv2.imshow('3',motion)
            out = np.concatenate((frame, motion, img_translation), axis=1)
            cv2.imwrite('output/'+str(frame_idx)+'.jpeg', out)
            cv2.imshow('out',out)
            cv2.waitKey(10)

        else:
            prev_frame = frame.copy()




def main():
	#test_image()
	test_video()

if __name__ == '__main__':
    main()

