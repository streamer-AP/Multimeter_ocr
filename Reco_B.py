import cv2
import os
from imutils import contours
import imutils
import numpy as np
from glob import glob

from imutils import contours
def digit_split(img):
    height, width = img.shape[:-1]
    img = img[:-height//8, :-width//10, :]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, _ = cv2.threshold(gray_img[10:-10,:], 50, 255, cv2.THRESH_OTSU)
    th,binary_img=cv2.threshold(gray_img,th,255,cv2.THRESH_BINARY)
    #binary_img = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,5)
    binary_img = 255 - binary_img

    # cnts = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL,
	# cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # for c in cnts:
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     if w*h < height * width / 200:
            # print(x,y,w,h)
            # binary_img = cv2.rectangle(binary_img, (x, y), (x+w, y+h), 0, thickness=-1)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3))
    # for i in range(1):
    #     binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_ELLIPSE, kernel, iterations=1)

    cnts = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cv2.findContours(binary_img.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    digitCnts = []
    digit_boxes = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        # print(x,y,w,h)
        if w < width/5 and w*h>100:
            digitCnts.append(c)
            digit_boxes.append((x,y,w,h))

    # print(digit_boxes)

    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

    return digitCnts, binary_img, digit_boxes


def check_minus(img, digit_y, digit_h):
    # img_copy = img.copy()
    check = False
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_OTSU)
    height, width = binary_img.shape
    binary_img = 255 - binary_img
    # plt.imshow(binary_img)
    top_left = (2, digit_y+digit_h//3)
    bottom_right = (width//6, digit_y+3*digit_h//4)

    mask = np.zeros_like(binary_img)
    mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 1
    binary_img = binary_img * mask
    # plt.imshow(binary_img)
    cnts = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # digitCnts = []
    minus_pos = None
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w*h > (bottom_right[1]-top_left[1])*(bottom_right[0]-top_left[0]) / 10:
            check = True
            minus_pos = [x,y,w,h]
        # print(x,y,w,h)
    # print(top_left, bottom_right)
    # img_copy = cv2.rectangle(img, top_left, bottom_right, color=(0,255,0), thickness=2)
    # plt.imshow(img_copy)
    return check, minus_pos


def check_dot(img, digit_boxes):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img[:-3, :]
    _, binary_img = cv2.threshold(gray_img, 20, 255, cv2.THRESH_BINARY_INV)
    # plt.imshow(binary_img)
    cnts = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    y_box = int(sum([box[1]+box[3] for box in digit_boxes])/len(digit_boxes))
    x_max = max([box[0] for box in digit_boxes])
    dot_region = None
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        # print(x,y,w,h)
        if 5 < w*h < 30 and abs(y-y_box) < 5 and x < x_max:
            # print(x,y,w,h)
            dot_region = (x,y, w, h)
            break
    return dot_region


def digit_recognition(img):
    height, width = img.shape[:-1]
    digitCnts, thresh, digit_boxes = digit_split(img)
    y = int(sum([box[1] for box in digit_boxes])/len(digit_boxes))
    h = int(sum([box[3] for box in digit_boxes])/len(digit_boxes))
    check, pos = check_minus(img, y, h)

    digits = []
    DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1): 0,
        (0, 0, 0, 0, 0, 0): 1,
        # (1, 0, 1, 1, 1, 0, 1): 1,
        (1, 0, 1, 1, 1, 0): 2,
        (1, 0, 1, 1, 0, 1): 3,
        (0, 1, 1, 1, 0, 1): 4,
        (1, 1, 0, 1, 0, 1): 5,
        (1, 1, 0, 1, 1, 1): 6,
        (1, 0, 1, 0, 0, 1): 7,
        (1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1): 9
    }


    for c in digitCnts:
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)
        roi = thresh[y:y + h, x:x + w]
        # compute the width and height of each of the 7 segments
        # we are going to examine
        (roiH, roiW) = roi.shape
        # plt.imshow(roi)
        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
        dHC = int(roiH * 0.05)
        dWC = int(roiH * 0.05)
        segments = [
            ((0, 0), (w, dH)),	# top
            ((2*dWC//3, dHC//2), (dW, h // 2)),	# top-left
            ((w - dW, 0), (w, h // 2)),	# top-right
            ((0, (h // 2) - dHC) , (w, (h // 2) + 4*dHC)), # center
            ((0, h // 2), (dW, h)),	# bottom-left
            ((w - int(1.5*dW), h // 2), (w-dWC, h-dHC)),	# bottom-right
            # ((0, h - dH), (w-dWC, h-dHC))	# bottom
        ]
        # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (127, 127, 0), (127, 0, 127), (0, 127, 127), (255, 255, 255)]
        on = [0] * len(segments)
        if w > width / 10:
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                # cv2.rectangle()
                # extract the segment ROI, count the total number of
                # thresholded pixels in the segment, and then compute
                # the area of the segment
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)
                # if the total number of non-zero pixels is greater than
                # 50% of the area, mark the segment as "on"
                if area>1 and total / float(area) > 0.5:
                    on[i]= 1
                # lookup the digit and draw it on the image
        try:
            digit = DIGITS_LOOKUP[tuple(on)]
        except:
            digit = 'error'
        digits.append(str(digit))
    return check, pos, digits, digit_boxes, thresh


def scan(img):
    height, width = img.shape[:-1]
    check, pos, digits, digit_boxes, thresh= digit_recognition(img)
    digit_boxes = sorted(digit_boxes, key=lambda x: x[0])
    dot_region = check_dot(img.copy(), digit_boxes)

    if dot_region is not None:
        x = dot_region[0]
        min_dis = 999
        min_dis_index = -1
        for i, digit_box in enumerate(digit_boxes):
            dis = abs(x - digit_box[0])
            if digits[i] == '1':
                dis = abs(x-(digit_box[0]-width/10))
            # print(dis)
            if dis < min_dis:
                min_dis = dis
                min_dis_index = i
        # print(min_dis, min_dis_index)
        # print(digit_boxes)
        digits.insert(min_dis_index, '.')
        img = cv2.rectangle(img, (dot_region[0], dot_region[1]), (dot_region[0]+dot_region[2], dot_region[1]+dot_region[3]), (255, 255, 0), thickness=1)

    if check:
        img = cv2.rectangle(img, (pos[0], pos[1]), (pos[0] + pos[2], pos[1] + pos[3]), (255, 0, 0), 2)

    height, width = img.shape[:-1]
    if check:
        digits = ['-'] + digits
    # print(str(digits))
    if 'error' in digits:
        print("error")
    for box in digit_boxes:
         img = cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (127, 127, 0), 2)

    cv2.putText(img, ''.join(digits), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return digits, img


if __name__ == '__main__':
    img_paths = sorted(glob('./panel/B/*'))
    save_root = './panel/B_final2'
    # img_paths = ['./panel/A/0006.jpg']
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for img_path in img_paths:
        digit, img_debug = scan(img_path)
        # print(digit)
        cv2.imwrite(os.path.join(save_root, img_path.split('/')[-1]), img_debug)

