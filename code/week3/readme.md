# Runing Mask RCNN for the first time
Run the model over all video frames and save the detections as pickle file. Low confidence used (0.5) as we can vary it as we want later wwithout running the model again , but 0.8 will produce less FP.


```
python mrcnn_detector.py --trained_model_path mask_rcnn_coco.h5 --video_dir ../datasets/AICity_data/train/S03/c010/vdo.avi --min_confidence 0.5 --save_dir rcnn_detections_before_tuning/
```

# Reading saved detections
This command will read saved detections and run visulize them for each frame. Note that detections saved for all categoris not cars only.

These two parameters can be adjusted: classes_of_interest, min_confidence

```
python3.6 visualize_detections.py
```