import cv2
import os
import numpy as np
from utils import  get_binary_img, get_contours, get_contours_bbox,digit_recognition,digit_split,check_dot

def draw_result(img,digits,digit_contours_bbox,minus_pos=None):
    if minus_pos is not None:
        img = cv2.rectangle(
            img, (minus_pos[0], minus_pos[1]), (minus_pos[0] + minus_pos[2], minus_pos[1] + minus_pos[3]), (255, 0, 0), 2)
    for box in digit_contours_bbox:
        img = cv2.rectangle(
        img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (127, 127, 0), 2)

    cv2.putText(img, ''.join(digits), (25, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return img

        
def binary_post_process(binary_img):
    height, width = binary_img.shape[0], binary_img.shape[1]
    contours = get_contours(binary_img.copy())
    contours_bbox = get_contours_bbox(contours)

    small_contours = [(x, y, w, h) for x, y, w, h in contours_bbox if w*h <
                      height * width / 100 or (y < height/10 and w*h < height*width/75)]
    for x, y, w, h in small_contours:
        binary_img = cv2.rectangle(
            binary_img, (x, y), (x+w, y+h), 0, thickness=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_img = cv2.dilate(binary_img, kernel, iterations=1)
    return binary_img

def scan(img):
    height, width = img.shape[:-1]
    binary_img = get_binary_img(img, method=cv2.THRESH_OTSU, low=50)
    top_roi = [slice(0, int(height*0.11)), slice(0, width)]
    mid_roi = [slice(int(height*0.10), int(height*0.667)), slice(0, width)]
    bottom_roi = [slice(int(height*0.877), int(height)), slice(0, width)]
    # top_roi = [slice(0, int(height*0.2)), slice(0, width)]
    # mid_roi = [slice(int(height*0.3), int(height*0.9)), slice(0, width)]
    # bottom_roi = [slice(int(height*0.9), int(height)), slice(0, width)]

    top_region = img[top_roi[0], top_roi[1]].copy()
    mid_region = img[mid_roi[0], mid_roi[1]].copy()
    mid_region_binary = binary_img[mid_roi[0], mid_roi[1]].copy()
    bottom_region = img[bottom_roi[0], bottom_roi[1]].copy()

    digit_contours_bbox,split_debug_img= digit_split(
        mid_region_binary.copy(), post_process=binary_post_process)
    if len(digit_contours_bbox) == 0:
        return [], mid_region_binary
    minus_pos, digits = digit_recognition(
        mid_region_binary.copy(), digit_contours_bbox, 0, 0)
    dot_idx = check_dot(mid_region_binary.copy(), digit_contours_bbox)
    if dot_idx != -1:
        digits.insert(dot_idx, '.')
    debug_img=draw_result(mid_region,digits,digit_contours_bbox,minus_pos)
    return digits, debug_img
