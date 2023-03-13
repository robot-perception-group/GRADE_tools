import numpy as np
import cv2 as cv

class Blur(object):
    def __init__(self, blur_fn, output_img_size, BLUR_IMG_FLAG):
        blur = np.load(blur_fn).item()
        self.blur_img_flag = BLUR_IMG_FLAG
        if self.blur_img_flag:
            self.img_w = 640
            self.img_h = 480
            self.output_img_w = output_img_size[0]
            self.output_img_h = output_img_size[1]
        else:
            self.img_w = output_img_size[0]
            self.img_h = output_img_size[1]
        self.Hs = blur['Hs']
        self.H_mean = blur['H_mean']
        self.exposure_time = blur['exposure_time']
        self.t_readout = blur['readout_time']
        self.interval = blur['interval']
        self.num_pose = blur['num_pose']
        self.extrinsic_mats = blur['extrinsic_mats']
        self.K = blur['intrinsic_mat']
    
    def blur_mask(self, masks):
        # consistent with the rosbag processing format
        if self.blur_img_flag:
            masks = cv.resize(masks, dsize=(self.img_w, self.img_h))
        
        masks = cv.warpPerspective(masks, self.H_mean, (self.img_w, self.img_h), flags=cv.INTER_LINEAR+cv.WARP_FILL_OUTLIERS, borderMode=cv.BORDER_REPLICATE)

        # add rolling shutter effect
        H_last = self.extrinsic_mats[-1,:]
        H_last = H_last.reshape((3,3))
        
        piece_H = int(1)
        y = piece_H  # y-1 is the row index

        new_pieces = []
        while y <= self.img_h:
            # time and approximated rotation for y th row
            t_y = self.t_readout * y / self.img_h
            H_y = self.interp_rot(t_y)
            H_new = np.matmul(H_y, np.linalg.inv(H_last))
            W_y = np.matmul(np.matmul(self.K, H_new), np.linalg.inv(self.K))

            old_piece = masks[y-piece_H:y, :]
            new_piece = cv.warpPerspective(old_piece, W_y, (self.img_w, piece_H), flags=cv.INTER_NEAREST+cv.WARP_FILL_OUTLIERS, borderMode=cv.BORDER_REPLICATE)
            new_pieces.append(new_piece)
            y += piece_H

        mask_blur_rs = np.concatenate(np.array(new_pieces), axis=0)
        
        if self.blur_img_flag:
            mask_blur_rs = cv.resize(mask_blur_rs, dsize=(self.output_img_w, self.output_img_h))
        
        return mask_blur_rs
    
    
    def blur_image(self, img):
        # Define the list of the images
        frames = []
        frames.append(img)

        for i in range(self.num_pose):
            h_mat = self.Hs[i]
            img_dst = cv.warpPerspective(img, h_mat, (self.img_w, self.img_h),flags=cv.INTER_LINEAR+cv.WARP_FILL_OUTLIERS, borderMode=cv.BORDER_REPLICATE)
            frames.append(img_dst)
            
        frames = np.array(frames)
        img_blur = np.mean(frames, axis=0)
        
        # add rolling shutter effect
        H_last = self.extrinsic_mats[-1,:]
        H_last = H_last.reshape((3,3))
        
        piece_H = int(1)
        y = piece_H  # y-1 is the row index

        new_pieces = []
        while y <= self.img_h:
            # time and approximated rotation for y th row
            t_y = self.t_readout * y / self.img_h
            H_y = self.interp_rot(t_y)
            H_new = np.matmul(H_y, np.linalg.inv(H_last))
            W_y = np.matmul(np.matmul(self.K, H_new), np.linalg.inv(self.K))

            old_piece = img_blur[y-piece_H:y, :, :]
            new_piece = cv.warpPerspective(old_piece, W_y, (self.img_w, piece_H), flags=cv.INTER_NEAREST+cv.WARP_FILL_OUTLIERS, borderMode=cv.BORDER_REPLICATE)
            new_pieces.append(new_piece)
            y += piece_H

        img_blur_rs = np.concatenate(np.array(new_pieces), axis=0)
        img_blur_rs = img_blur_rs.astype(np.uint8)
        
        return img_blur_rs
    
    
    def interp_rot(self, t):
        h_array = self.extrinsic_mats
        exposure_ts= np.array([i * self.interval for i in range(self.num_pose+1)])
        if t >= exposure_ts[-1]:
            H_last = h_array[-1, :]
            return H_last.reshape((3,3))

        rot_t = np.array([0.]*9)
        for i in range(9):
            rot_t[i] = np.interp(t, exposure_ts, h_array[:, i])

        return rot_t.reshape((3, 3))