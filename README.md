
![](images/logo.png?raw=true)


# Video Surveillance for Road Traffic Monitoring

 This repository contains information about the work done during the project of Module 6 of [Master in Computer Vision](http://pagines.uab.cat/mcv/) 2018.

 ## Usage

 Each week's directory contains the code with implementation and data to test the tasks and compute the scores. In [annotations](annotations) directory the are .xml files containing annotations to cars and bikes (along with unique ids) to the video sequence used during the project.

 ### Install dependencies
Before trying the code run
```
pip3 install -r requirements.txt
```
Python 3.5 or higher is needed.


## Week 1 Video annotation and scoring metrics
During the first week of the project, the main tasks were to prepare the annotations of data and metric computation tools to be able to evaluate the results during the project.
Tool used for video annotation: [CVAT](https://github.com/opencv/cvat)
#### Implemented metrics:
 - IoU
 - mAP
 - MSEN
 - PEPN 
## Week 2 Background estimation
#### Gaussian model static/adaptive comparison
Each pixel is modelled by a gaussian with its
own mean and variance.
We used the first 25% of the video sequence
to calculate both mean and variance for each
pixel.
For the left 75% we consider foreground the
pixels with big variations with respect its mean
and variance and background the ones that
barely change.
#### Comparison between static and adaptive background estimation:
![](images/gauss_adaptive.png?raw=true)

## Week 3 Object detection and tracking
We used Mask-RCNN tensorflow implementation from https://github.com/matterport/Mask_RCNN
We used the pretrained model on COCO and  considered these classes:
['car','bus','truck','bicycle', 'motorcycle']
![](images/segmented.png?raw=true)

The model trained on COCO was fine-tuned to the sequence used in the task.
It allowed to improve the scores (tested with IoU>0.5):

| Model | mAP |
|-|-|
|Off-the-shelf|0.416|
|Fine-tuned|0.461|

The maximum overlap tracking and Kalman filter were tested:

| Method | IDF1 |
|-|-|
|Max-overlap|0.686|
|Kalman-filter|0.612|

## Week 4 Optical flow, video stabilisation, tracking
### 
#### Implement optical flow using block-matching
![](images/opt_flow1.png?raw=true)

The score were computed for frames from the KITTI dataset. It was discovered that the axes in optical flow computation should be swapped to properly fit the ground truth data.

| Axes order | Avg. PEPN | Avg. MSEN|
|-|-|-|
|Normal|0.79|11.58|
|Swapped|0.71|9.93|

#### Video stabilisation using block-matching
![](images/stabilisation.png?raw=true)

<p align="center">
 <img width="720" height="180" src=images/1.gif?raw=true>
</p>

#### Using optical flow to improve performance of tracking:
The idea was to combine maximum overlap tracking algorithm with velocity vector computed for each detection. This approach allows to distinguish overlapping objects moving in different directions.
##### Optical flow computed for tested sequence:
![](images/opt_flow_cars.gif?raw=true)

## Week 5 Multi-target camera tracking
#### Multi-target single camera tracking (MTSC)
#### Multi-target single camera tracking (MTMC)

### Authors
- Basem Elbarashy
- Sergi Garcia
- Manuel Rey Area
- Grzegorz Skorupko
