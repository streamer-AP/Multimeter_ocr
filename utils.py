import cv2
import imutils
import numpy as np
import torch
from PIL import  Image
from torchvision.transforms import ToTensor
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}


def color_transfer(src,dst):
    src=cv2.cvtColor(src,cv2.COLOR_BGR2LAB)
    dst=cv2.cvtColor(dst,cv2.COLOR_BGR2LAB)

    src_mean,src_std=cv2.meanStdDev(src)
    dst_mean,dst_std=cv2.meanStdDev(dst)
    src,dst=src.astype(np.float32),dst.astype(np.float32)
    src=(src-src_mean)*(dst_std/src_std)+dst_mean


def get_binary_img(img, method=cv2.THRESH_OTSU, inv=True, low=50, high=255):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, low, high, method)
    if inv == True:
        binary_img = 255 - binary_img
    return binary_img


def get_contours(binary_img):
    contours = cv2.findContours(
        binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return contours


def get_contours_bbox(contours):
    return [cv2.boundingRect(c) for c in contours]


def get_segment(roiW, roiH):
    (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
    dHC = int(roiH * 0.05)
    dWC = int(roiH * 0.05)
    return [
        ((0, 0), (roiW, dH)),  # top
        ((2*dWC//3, dHC//2), (dW+dWC, roiH // 2)),  # top-left
        ((roiW - dW, 0), (roiW, roiH // 2)),  # top-right
        ((0, (roiH // 2) - dHC), (roiW, (roiH // 2) + dHC)),  # center
        ((0, roiH // 2), (dW, roiH)),  # bottom-left
        ((roiW - int(1.5*dW), roiH // 2), (roiW-dWC, roiH-dHC)),  # bottom-right
        ((0, roiH - dH), (roiW-dWC, roiH-dHC))  # bottom
    ]

def check_minus(binary_img, digit_top, digit_height):
    check = False
    width = binary_img.shape[1]

    roi = binary_img[digit_top+digit_height//3:digit_top +
                     3*digit_height//4, width//24:width//6]
    roi_area = roi.shape[0]*roi.shape[1]
    contours_bbox = get_contours_bbox(roi)
    minus_pos = None
    for (x, y, w, h) in contours_bbox:
        if w*h > roi_area / 10:
            check = True
            minus_pos = [x, y, w, h]
    return check, minus_pos
def get_digit_imgs(img,top_ratio,bottom_ratio,pre_contour_box,pre_char_num,pre_char_width,pre_first_char_offset=20):
    height, width = img.shape[:-1]
    mid_roi = [slice(int(height*top_ratio), int(height*bottom_ratio)), slice(0, width)]

    mid_region = img[mid_roi[0], mid_roi[1]].copy()
    mid_region_gray = cv2.cvtColor(mid_region, cv2.COLOR_BGR2GRAY)

    digit_contours_bbox = digit_split(mid_region_gray,pre_first_char_offset,pre_char_num,pre_char_width,pre_contour_box=pre_contour_box)
    digit_imgs=[mid_region[v[1]:v[1]+v[3],v[0]:v[0]+v[2]] for v in digit_contours_bbox ]

    return digit_imgs
def get_digits(imgs,model,char_dict):
    digits=[]
    with torch.no_grad():
        for img in imgs:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            input_tensor=ToTensor(cv2.resize(img,(40,80))).unsqueeze(0)
            output=model(input_tensor).detach().cpu().numpy()
            digits.append(char_dict[np.argmax(output[0])])
    return digits

def get_unit(img,model,transform,char_dict):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=Image.fromarray(img)
    input_tensor=transform(img)
    with torch.no_grad():
        output=model(input_tensor.unsqueeze(0)).detach().cpu().numpy()
        print(output)
    return char_dict[np.argmax(output[0])]


def digit_split(gray_img,pre_first_char_offset=20, char_num=4, pre_char_width=50, post_process=None, pre_contour_box=None):
    height,width=gray_img.shape[0],gray_img.shape[1]
    vertical_projection = np.sum(255-gray_img, axis=0)
    min_background_area = np.Inf
    best_char_width = pre_char_width
    best_first_char_offset = pre_first_char_offset
    for char_width in range(max(0,pre_char_width-10), pre_char_width+10):
        for first_char_offset in range(max(0,pre_first_char_offset-20),pre_first_char_offset):
            if char_num*char_width+first_char_offset < width:
                background_area = np.sum(
                    vertical_projection[first_char_offset::char_width])
                if background_area < min_background_area:
                    min_background_area = background_area
                    best_char_width = char_width
                    best_first_char_offset = first_char_offset

    contours_bbox = [(max(0,best_first_char_offset+int(best_char_width*(idx-0.2))), 0,
                          int(best_char_width*1.4),height) for idx in range(char_num)]
    return contours_bbox

def check_dot(binary_img, digit_contours_bbox,max_dot_digit_dist=20,dot_area_min=20,dot_area_max=120, debug=False):
    contours = get_contours(binary_img)
    contours_bbox = get_contours_bbox(contours)
    digit_bottom = int(
        sum([box[1]+box[3] for box in digit_contours_bbox])/len(digit_contours_bbox))
    digit_right = max([box[0] for box in digit_contours_bbox])
    dot_x = -1
    best_dist = max_dot_digit_dist
    contour_box = None
    for (x, y, w, h) in contours_bbox:
        if dot_area_min < w*h < dot_area_max and abs(y+h-digit_bottom) < best_dist and x < digit_right:
            best_dist = abs(y+h-digit_bottom)
            dot_x = x
            contour_box = (x,y,w,h) 
    dot_idx = -1
    if dot_x != -1:
        for idx, bbox in enumerate(digit_contours_bbox):
            if abs(bbox[0]-dot_x) < abs(digit_contours_bbox[dot_idx][0]-dot_x):
                dot_idx = idx
    if debug:
        return dot_idx, contour_box
    else:
        return dot_idx
