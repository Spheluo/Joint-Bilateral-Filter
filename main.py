import numpy as np
import pandas as pd
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter

def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()
    
    ### TODO ###
    
    with open(args.setting_path) as f:
        setting = [i.rstrip('\n').split(',') for i in f.readlines()]
        RGB_setting = setting[1:6]
        sigma_s, sigma_r = int(setting[6][1]), float(setting[6][3])

    img = cv2.imread(args.image_path) # read 1.png
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #convert to RGB
    JBF = Joint_bilateral_filter(2, 0.1) # create object for JBF
    cost = {}
    # initial cv2 gray conversion
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb) # bf
    jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray) #JBF (gray as guidance)
    # calculate cost by L1 normalization
    cost['cv2.COLOR_BGR2GRAY'] = np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32')))
    # save figures
    jbf_out = cv2.cvtColor(jbf_out, cv2.COLOR_BGR2RGB) # convert to BGR
    cv2.imwrite(args.image_path[:-4]+'_img_gray_1_cv2_COLOR_BGR2GRAY.png', img_gray)
    cv2.imwrite(args.image_path[:-4]+'_jbf_1_cv2_COLOR_BGR2GRAY.png', jbf_out)
    # test different combination of RGB
    for r, g, b in RGB_setting:
        #convert to gray
        img_gray = img_rgb[:,:,0]*float(r)+img_rgb[:,:,1]*float(g)+img_rgb[:,:,2]*float(b)
        jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray) #JBF
        cost[f'R*{r}+G*{g}+B*{b}'] = np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32')))
        jbf_out = cv2.cvtColor(jbf_out,cv2.COLOR_BGR2RGB) # convert to BGR
        cv2.imwrite(args.image_path[:-4]+f'_jbf_1_{r}_{g}_{b}.png', jbf_out) #export
        cv2.imwrite(args.image_path[:-4]+f'_img_gray_1_{r}_{g}_{b}.png', img_gray)
    # output the cost matrix
    pd.Series(cost, name='Cost(1.png)').to_excel(args.image_path[:-4]+'_costs.xlsx')

if __name__ == '__main__':
    main()