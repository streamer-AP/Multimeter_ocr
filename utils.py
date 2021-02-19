import cv2
import imutils
import numpy as np

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

def digit_recognition(binary_img, digit_contours_bbox, x_offset=0, y_offset=0):
    width = binary_img.shape[1]
    digit_top = int(
        sum([box[1] for box in digit_contours_bbox])/len(digit_contours_bbox))
    digit_height = int(
        sum([box[3] for box in digit_contours_bbox])/len(digit_contours_bbox))
    digits = []
    check, pos = check_minus(binary_img, digit_top, digit_height)
    if check:
        digits.append("-")
    for (x, y, w, h) in digit_contours_bbox:
        x = x+x_offset
        y = y+y_offset
        roi = binary_img[y:y + h, x:x + w]
        segments = get_segment(w, h)
        on = [0] * len(segments)
        if w > 0.1*width:
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                yB = max(yA+1, yB)
                xB = max(xA+1, xB)
                segROI = roi[yA:yB, xA:xB]
                char_area = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)
                if (char_area+0.0001) / (float(area)+0.001) > 0.4:
                    on[i] = 1
        digit = DIGITS_LOOKUP.get(tuple(on), "unkwon")
        digits.append(str(digit))
    return pos, digits


def digit_split(binary_img, pre_first_char_offset=20, char_num=4, pre_char_width=50, post_process=None):

    if post_process != None:
        binary_img = post_process(binary_img)
    height, width = binary_img.shape[0], binary_img.shape[1]
    contours = get_contours(binary_img.copy())
    contours_bbox = get_contours_bbox(contours)

    vertical_projection = np.sum(binary_img, axis=0)
    min_background_area = np.Inf
    best_char_width = pre_char_width
    best_first_char_offset = pre_first_char_offset
    for char_width in range(pre_char_width-5, pre_char_width+5):
        for first_char_offset in range(pre_first_char_offset):
            if char_num*char_width+first_char_offset < width:
                background_area = np.sum(
                    vertical_projection[first_char_offset::char_width])
                if background_area < min_background_area:
                    min_background_area = background_area
                    best_char_width = char_width
                    best_first_char_offset = first_char_offset
    avg_digit_top = 0
    avg_digit_height = 0
    visable_digit = 0
    for (x, y, w, h), contour in zip(contours_bbox, contours):
        if y < height*0.25 and w < width*0.25 and h > height*0.5 and w > width*0.05:
            avg_digit_top += y
            avg_digit_height += h
            visable_digit += 1
    if visable_digit > 0:
        avg_digit_height /= visable_digit
        avg_digit_top /= visable_digit
        contours_bbox = [(best_first_char_offset+best_char_width*idx, int(avg_digit_top),
                          best_char_width, int(avg_digit_height)) for idx in range(char_num)]
    else:
        contours_bbox = []
    return contours_bbox, binary_img

def check_dot(binary_img, digit_contours_bbox,max_dot_digit_dist=20,dot_area_min=20,dot_area_max=120):
    contours = get_contours(binary_img)
    contours_bbox = get_contours_bbox(contours)
    digit_bottom = int(
        sum([box[1]+box[3] for box in digit_contours_bbox])/len(digit_contours_bbox))
    digit_right = max([box[0] for box in digit_contours_bbox])
    dot_x = -1
    best_dist = max_dot_digit_dist
    for (x, y, w, h) in contours_bbox:
        if dot_area_min < w*h < dot_area_max and abs(y+h-digit_bottom) < best_dist and x < digit_right:
            best_dist = abs(y+h-digit_bottom)
            dot_x = x
    dot_idx = 0
    if dot_x != -1:
        for idx, bbox in enumerate(digit_contours_bbox):
            if abs(bbox[0]-dot_x) < abs(digit_contours_bbox[dot_idx][0]-dot_x):
                dot_idx = idx
    return dot_idx
