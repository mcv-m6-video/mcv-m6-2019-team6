import matplotlib.pyplot as plt
import numpy as np
import os 
import cv2

LKflow_45  = 'LKflow_000045_10.png'
LKflow_157 = 'LKflow_000157_10.png'
gt_45      = '000045_10.png'
gt_157     = '000157_10.png'
gray_45    = 'gray_000045_10.png'
gray_157   = 'gray_000157_10.png'

def read_file(path):
    """Read an optical flow map from disk
    Optical flow maps are stored in disk as 3-channel uint16 PNG images,
    following the method described in the KITTI optical flow dataset 2012
    (http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow).
    Returns:
      numpy array with shape [height, width, 3]. The first and second channels
      denote the corresponding optical flow 2D vector (u, v). The third channel
      is a mask denoting if an optical flow 2D vector exists for that pixel.
      Vector components u and v values range [-512..512].
    """
    data = cv2.imread(path, -1).astype('float32')
    result = np.empty(data.shape, dtype='float32')
    result[:,:,0] = (data[:,:,2] - 2**15) / 64
    result[:,:,1] = (data[:,:,1] - 2**15) / 64
    result[:,:,2] = data[:,:,0]

    return result


def flow_metrics(pred, gt):

    flowExist  = (gt[:,:,2] == 1)
    pred_flow  = pred[flowExist]
    gt_flow    = gt[flowExist]
    #print(flowExist.shape)
    img_err = np.zeros(shape=gt[:,:,1].shape)
    
    err = gt_flow[:,:2] - pred_flow[:,:2]

    squared_err = np.sum(err**2, axis=1)
    vect_err = np.sqrt(squared_err)
    hit = vect_err < 3.0
    img_err[flowExist] = vect_err

    msen = np.mean(vect_err)
    pepn = 100 * (1 - np.mean(hit))

    return msen, pepn, img_err, vect_err


def analyze_flow(pred_im, gt_im, gray_im, seq_no):
    pred = read_file(pred_im)
    gt   = read_file(gt_im)
    gray = cv2.imread(gray_im)
    # ================================= error calculation
    msen, pepn, img_err, vect_err = flow_metrics(pred,gt)

    print('Sequence: ', seq_no)
    print('PEPN = ', pepn)
    print('MSEN =  ', msen)

    # =================================   hist plot
    plt.figure(1)
    plt.title('hist of vect_err')
    plt.hist(vect_err, bins=25, normed=1,stacked=False)
    plt.show()    
    # ================================= plotting error per pixel (image)
    plt.figure(2) #print(img_err.shape)
    plt.imshow(img_err)
    plt.colorbar()
    plt.show()
    # ================================= better error visualization
    tmp = gray.copy()
    img_err = img_err / img_err.max() * 255
    tmp[:,:,0] = img_err
    tmp[:,:,1] = img_err
    tmp[:,:,2] = img_err
    err_color = cv2.applyColorMap(tmp, cv2.COLORMAP_JET)
    dst = cv2.addWeighted(err_color,0.5,gray,0.5,0)
    print(gray.shape)
    print(tmp.shape)
    cv2.imshow('dst',dst)
    cv2.waitKey(1000)




if __name__ == '__main__':
    analyze_flow(LKflow_45, gt_45, gray_45, '45')
    analyze_flow(LKflow_157, gt_157, gray_157, '157')


