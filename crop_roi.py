import cv2
from glob import glob
import os
import numpy as np

# data_root = './data/Reco3_output'
# cats = os.listdir(data_root)
# output_root = './data/Reco3_crop'

# for cat in cats:
#     img_paths = glob(os.path.join(data_root, cat, '*'))
#     save_dir = os.path.join(output_root, cat)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     for img_path in img_paths:
#         img = cv2.imread(img_path)
#         height, width = img.shape[:-1]
#         # top_roi = [slice(0, int(height * 0.3)), slice(0, width)]
#         # bottom_roi = [slice(int(height * 0.6), int(height)), slice(0, width)]
#         # top_roi_crop = img[top_roi[0], top_roi[1]].copy()
#         # bottom_roi_crop = img[bottom_roi[0], bottom_roi[1]].copy()
#         # crop_img = np.concatenate([top_roi_crop, bottom_roi_crop], axis=0)
#         side_roi = [slice(0, height), slice(int(width*0.87), width)]
#         side_roi_crop = img[side_roi[0], side_roi[1]].copy()
#         cv2.imwrite(os.path.join(save_dir, img_path.split('/')[-1]), side_roi_crop)

data_root = './data/Reco1_output'
cats = os.listdir(data_root)
output_root = './data/Reco1_crop_2'

for cat in cats:
    img_paths = glob(os.path.join(data_root, cat, '*'))
    save_dir = os.path.join(output_root, cat)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_path in img_paths:
        img = cv2.imread(img_path)
        height, width = img.shape[:-1]
        top_roi = [slice(0, int(height * 0.11)), slice(0, width)]
        bottom_roi = [slice(int(height * 0.6), int(height)), slice(0, width)]
        top_roi_crop = img[top_roi[0], top_roi[1]].copy()
        bottom_roi_crop = img[bottom_roi[0], bottom_roi[1]].copy()
        crop_img = np.concatenate([top_roi_crop, bottom_roi_crop], axis=0)
        cv2.imwrite(os.path.join(save_dir, img_path.split('/')[-1]), crop_img)



# data_root_side = './data/Reco3_crop_side'
# data_root_top = './data/Reco3_crop_top'
# output_root = './data/Reco3_crop'
#
# cats = os.listdir(data_root_side)
#
# for cat in cats:
#     img_paths_top = glob(os.path.join(data_root_top, cat, '*'))
#     img_paths_side = glob(os.path.join(data_root_side, cat, '*'))
#     save_dir = os.path.join(output_root, cat)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     for i in range(len(img_paths_side)):
#         img_top = cv2.imread(img_paths_top[i])
#         img_side = cv2.imread(img_paths_side[i])
#         # print(img_side.shape)
#         img_side = cv2.resize(img_side, (img_side.shape[1], img_top.shape[1]))
#         img_side = np.transpose(img_side, (1,0,2))
#
#
#         # img_side.resize(img_side.shape[0], img_top.shape[1])
#         img_cat = np.concatenate([img_top, img_side], axis=0)
#         cv2.imwrite(os.path.join(save_dir, img_paths_top[i].split('/')[-1]), img_cat)
