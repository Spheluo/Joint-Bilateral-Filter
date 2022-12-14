import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, 
                                        self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, 
                                             self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        # setup a look-up table for spatial kernel
        LUT_s = np.exp(-0.5*(np.arange(self.pad_w+1)**2)/self.sigma_s**2)
        # setup a look-up table for range kernel
        LUT_r = np.exp(-0.5*(np.arange(256)/255)**2/self.sigma_r**2)
        # compute the weight of range kernel by rolling the whole image
        wgt_sum, result = np.zeros(padded_img.shape), np.zeros(padded_img.shape)
        for x in range(-self.pad_w, self.pad_w+1):
            for y in range(-self.pad_w, self.pad_w+1):
                # method 1 (easier but slower)
                dT = LUT_r[np.abs(np.roll(padded_guidance, [y,x], axis=[0,1])-padded_guidance)]
                r_w = dT if dT.ndim==2 else np.prod(dT,axis=2) # range kernel weight
                s_w = LUT_s[np.abs(x)]*LUT_s[np.abs(y)]#spatial kernel
                t_w = s_w*r_w
                padded_img_roll = np.roll(padded_img, [y,x], axis=[0,1])
                for channel in range(padded_img.ndim):
                    result[:,:,channel] += padded_img_roll[:,:,channel]*t_w
                    wgt_sum[:,:,channel] += t_w
        output = (result/wgt_sum)[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w,:]
        
        return np.clip(output, 0, 255).astype(np.uint8)