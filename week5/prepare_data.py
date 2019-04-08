import os
import cv2
import numpy as np

path_sequences = 'aic19-track1-mtmc-train/'
train_folders_path = 'aic19-track1-mtmc-train/siamese_train/'
ids_array = np.zeros(1000, dtype=np.uint)

def save_detections(path_video, gt_file, roi):
    with open(gt_file) as f:
        data = f.readlines()

    success = True
    video_cap = cv2.VideoCapture(path_video)
    frame_idx = 0
    while True:
        success, frame = video_cap.read()
        frame_idx += 1
        if not success:
            break

        for line in data:
            splits = line.split(",")
            id = splits[1]
            if str(frame_idx) == splits[0]:
                path_to_save = os.path.join(train_folders_path, str(id))
                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)
                y = int(splits[3])
                h = int(splits[5])
                x = int(splits[2])
                w = int(splits[4])
                crop_img = frame[y:y+h, x:x+w]
                filename = path_to_save+"/crop_%05d.png" % ids_array[int(id)]
                cv2.imwrite(filename, crop_img)
                cv2.imshow("cropped", crop_img)
                cv2.waitKey(0)
                ids_array[int(id)] += 1


if __name__ == "__main__":
    if not os.path.exists(path_sequences):
        print('There is not such folder %s' % path_sequences)
        exit(0)

    if not os.path.exists(train_folders_path):
        os.makedirs(train_folders_path)

    for root, dirs, files in os.walk(path_sequences, topdown=True):
        for name in dirs:
            current_dir = os.path.join(root,name)
            if os.path.isdir(current_dir) and ('s01' in current_dir.lower() or 's04' in current_dir.lower()):
                if 'gt' in current_dir.lower():
                    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
                    gt_file_path = None
                    for folder in os.listdir(parent_dir):
                        if os.path.isdir(os.path.join(parent_dir, folder)) and folder.lower() == 'gt':
                            _, _, gt_file = os.walk(os.path.join(parent_dir, folder)).__next__()
                            gt_file_path = parent_dir + os.sep + folder + os.sep + gt_file[0]

                    (dirpath, dirnames, filenames) = os.walk(parent_dir).__next__()

                    # Filenames = roi.jpg, vdo.avi, calibration.txt

                    save_detections(os.path.join(parent_dir, filenames[1]), gt_file_path, os.path.join(parent_dir,
                                                                                                       filenames[0]))