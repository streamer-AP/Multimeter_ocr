# %%
import os
from glob import glob

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

%matplotlib inline
img_name_list = glob("DYB/*.jpg")

img = cv2.imread('DYB/WIN_20210123_13_43_27_Pro.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.medianBlur(img_gray, 5)

circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 50,
                           param1=10, param2=150, minRadius=50, maxRadius=215)


circles = np.uint16(np.around(circles))
print(circles)
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
plt.imshow(img)
plt.show()
# %%
print(img)
# %%
print(circles)
# %%
print(len(circles[0]))
# %%


def detect_circle(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 5)
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=10, param2=150, minRadius=50, maxRadius=215)
    circles = np.uint16(np.around(circles))
    mask = np.zeros_like(img_gray)
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv2.circle(mask, (i[0], i[1]), int(170*0.7), (1,), thickness=-1)
    return img, circles, mask

img_name_list = glob("DYB/*.jpg")
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

panel_dir = "panel"
if not os.path.exists(panel_dir):
    os.makedirs(panel_dir)

not_detected_num = 0
for img_path in tqdm(img_name_list):
    img = cv2.imread(img_path)
    # try:
    detected_img, detected_circles, mask = detect_circle(img)
    cv2.imwrite(os.path.join(
        output_dir, os.path.basename(img_path)), detected_img)
    img_panel = img * mask[..., None]
    cv2.imwrite(os.path.join(
        panel_dir, os.path.basename(img_path)), img_panel)
    # except:
    # not_detected_num+=1
    # print(f"not detected {img_path}, {not_detected_num}")
# %%
img_name_list = glob("panel/*.jpg")
panel = cv2.imread(img_name_list[1])
panel_gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
panel_gray = cv2.equalizeHist(panel_gray)
panel_gray = cv2.medianBlur(panel_gray, 5)
panel_binary = cv2.adaptiveThreshold(panel_gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=21, C=16)
panel_binary = cv2.medianBlur(panel_binary, 5)
# edges = cv2.Canny(panel_gray, 120, 150, apertureSize=3)

# plt.imshow(panel_binary)

b_hist = plt.hist(panel_gray.ravel(), bins=50, color='b',range=(1,255))
plt.show()
'''
plt.hist(range(0,180),hist)
lines = cv2.HoughLinesP(panel_binary, 1, np.pi/180, 60, minLineLength=200, maxLineGap=20)
print(lines)
for line in lines:
    # print(type(line))
    x1, y1, x2, y2 = line[0]
    cv2.line(panel, (x1, y1), (x2, y2), (255, 0, 0), 2)

plt.imshow(panel)
plt.show()
'''


# %%
plt.imshow(panel_gray,cmap="gray")
plt.show()
# %%
plt.imshow(panel)

plt.show()
# %%
img=cv2.imread(img_name_list[1])
cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(panel)
plt.show()

# %%
print(img_name_list[1])
# %%
import cv2
import numpy as np
from glob import glob
import os
from matplotlib import pyplot as plt

dir_path="output_1"
img_path_list=glob(os.path.join(dir_path,"*.jpg"))
img=cv2.imread(img_path_list[0])
img=img[50:350,250:550]
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(img,cmap="gray")
plt.show()
#%%
ret,thresh1 = cv2.threshold(img,40,255,cv2.THRESH_BINARY)
plt.imshow(thresh1,cmap="binary")
plt.show()
# %%
b_hist = plt.hist(img.ravel(), bins=50, color='b',range=(10,150))
plt.show()
# %%
x_sum=np.sum(255-thresh1,axis=1)
y_sum=np.sum(255-thresh1,axis=0)
# %%
plt.plot(x_sum)
# %%
plt.plot(y_sum)
# %%
x_sum_1=x_sum[1:]
x_sum_2=x_sum[:-1]
plt.plot(x_sum_1-x_sum_2)
# %%
from scipy.signal import convolve
kernel = np.array([1,0,1])
plt.plot(convolve(x_sum, kernel))#一维卷积运算
# %%
cnt_up=1
sigma_x=10
x_sum_up=np.zeros_like(x_sum)
for i in range(1,x_sum.shape[0]):
    if x_sum[i]<x_sum[i-1]-sigma_x:
        cnt_up+=1
        x_sum_up[i]=cnt_up
    else:
        cnt_up=0
    
plt.plot(x_sum_up)
x_1=np.argmax(x_sum_up)
x_2=np.argmax(x_sum_up[x_1+1:])
print(x_1,x_2+x_1)
# %%
cnt_up=1
sigma_x=0
y_sum_up=np.zeros_like(y_sum)
for i in range(1,y_sum.shape[0]):
    if y_sum[i-1]>y_sum[i]-sigma_x:
        cnt_up+=y_sum[i-1]-y_sum[i]
        y_sum_up[i]=cnt_up
    else:
        cnt_up=0
    
plt.plot(y_sum_up)
y_1=np.argmax(y_sum_up)
y_2=np.argmax(y_sum)
print(y_1,y_2)
# %%
plt.plot(y_sum)
# %%
img=cv2.imread(img_path_list[0])
img=img[50:350,250:550]
B,G,R=cv2.split(img)
# %%
img_mask=np.abs(B>np.clip(R.astype(np.int)+70,0,255)).astype(np.uint8)
img_mask=cv2.medianBlur(img_mask,3)
plt.imshow(np.abs(img_mask),cmap="gray")
plt.show()
circles = cv2.HoughCircles(img_mask, cv2.HOUGH_GRADIENT, 1, 5,
                           param1=2, param2=10, minRadius=3, maxRadius=100)
print(circles)

# %%
# %%
img_edge=cv2.Canny(img_mask,1,2)
plt.imshow(img_edge)
plt.show()
# %%
def panel_crop(img,roi=[70,370,250,550]):
    img=img[roi[0]:roi[1],roi[2]:roi[3]]
    height,width=img.shape[0],img.shape[1]
    B,G,R=cv2.split(img)
    img_mask_R=np.abs(R>np.clip(B.astype(np.int)+70,0,255)).astype(np.uint8)
    img_mask_B=np.abs(B>np.clip(R.astype(np.int)+100,0,255)).astype(np.uint8)
    edge_R=cv2.Canny(img_mask_R,1,2)
    edge_B=cv2.Canny(img_mask_B,1,2)

    circles_R = cv2.HoughCircles(img_mask_R, cv2.HOUGH_GRADIENT, 1, 10,
                           param1=2, param2=10, minRadius=5, maxRadius=50)
    circles_B=cv2.HoughCircles(img_mask_B, cv2.HOUGH_GRADIENT, 1, 10,
                           param1=2, param2=10, minRadius=5, maxRadius=50)
    y_1,y_2,x_1,x_2=0,height,0,width
    if len(circles_B)>0 and len(circles_R)>0:
        x_R=circles_R[0][0][0]
        y_R=circles_R[0][0][1]
        x_B=circles_B[0][0][0]
        y_B=circles_B[0][0][1]
        if x_R<width//2:
            img=cv2.transpose(img)
            img=cv2.flip(img,0)
        else:
            img=cv2.transpose(img)
        x_R,y_R,x_B,y_B=y_R,height-x_R,y_B,height-x_B
        degree=np.math.atan2(y_B-y_R,x_B-x_R)/np.math.pi*180
        M = cv2.getRotationMatrix2D((x_R,y_R), degree, 1.0)
        img = cv2.warpAffine(img, M, (width, height))
        dist=np.math.sqrt((x_R-x_B)**2+(y_R-y_B)**2)
        x_1,y_1=int(x_R-dist*0.1),int(y_R-dist*1.3)
        x_2,y_2=int(x_R+dist*1.12),int(y_R-dist*0.15)

    return img[y_1:y_2,x_1:x_2],img,edge_B,edge_R
# %%
from tqdm import tqdm
import shutil
from glob import glob
import cv2
import numpy as np
output_dir="panel1"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
img_path_list=glob("output_1/*.jpg")
img_path_list=sorted(img_path_list)
for img_path in tqdm(img_path_list[:86]):
    img=cv2.imread(img_path)
    img_crop,img,edge_B,edge_R=panel_crop(img)
    try:
        cv2.imwrite(os.path.join(output_dir,os.path.basename(img_path)),img_crop)
    except:
        print(img_path)

# %%
import cv2
class panel_cropper():
    
def panel_crop_debug(img,roi=[70,370,250,550]):
    img=img[roi[0]:roi[1],roi[2]:roi[3]]
    height,width=img.shape[0],img.shape[1]
    B,G,R=cv2.split(img)
    img_mask_R=np.abs(R>np.clip(B.astype(np.int)+70,0,255)).astype(np.uint8)
    img_mask_B=np.abs(B>np.clip(R.astype(np.int)+100,0,255)).astype(np.uint8)
    edge_R=cv2.Canny(img_mask_R,1,2)
    edge_B=cv2.Canny(img_mask_B,1,2)

    circles_R = cv2.HoughCircles(img_mask_R, cv2.HOUGH_GRADIENT, 1, 10,
                           param1=2, param2=10, minRadius=5, maxRadius=50)
    circles_B=cv2.HoughCircles(img_mask_B, cv2.HOUGH_GRADIENT, 1, 10,
                           param1=2, param2=10, minRadius=5, maxRadius=50)
    y_1,y_2,x_1,x_2=0,height,0,width
    if len(circles_B)>0 and len(circles_R)>0:
        for circle_R in circles_R[0]:
            x_R=circle_R[0]
            y_R=circle_R[1]
            r_R=circle_R[2]
            for circle_B in circles_B[0]:
                x_B=circle_B[0]
                y_B=circle_B[1]
                r_B=circle_B[2]
                prob=abs(r_B-r_R)+abs(r_B)
        if x_R<width//2:
            img=cv2.transpose(img)
            img=cv2.flip(img,0)
        else:
            img=cv2.transpose(img)
        x_R,y_R,x_B,y_B=y_R,height-x_R,y_B,height-x_B
        degree=np.math.atan2(y_B-y_R,x_B-x_R)/np.math.pi*180
        M = cv2.getRotationMatrix2D((x_R,y_R), degree, 1.0)
        img = cv2.warpAffine(img, M, (width, height))
        dist=np.math.sqrt((x_R-x_B)**2+(y_R-y_B)**2)
        x_1,y_1=int(x_R-dist*0.1),int(y_R-dist*1.3)
        x_2,y_2=int(x_R+dist*1.12),int(y_R-dist*0.15)

    return img[y_1:y_2,x_1:x_2],img,edge_B,edge_R,circles_B,circles_R
test_img_path="output_1/0014.jpg"
img=cv2.imread(test_img_path)
img_out,img_crop,edge_B,edge_R,circles_B,circles_R=panel_crop_debug(img)
# %%
from matplotlib import pyplot as plt
plt.imshow(edge_R)
# %%
import cv2
from matplotlib import pyplot as plt
import numpy as np
#%%
img=cv2.imread("output_1/0150.jpg")
plt.imshow(img)
#%%
def point_distance_line(point,line_point1,line_point2):
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_point1-line_point2)
    return distance
def cropper(img,roi=[100,400,200,500]):
    width,height=img.shape[1],img.shape[0]
    B,G,R=cv2.split(img)
    img_roi=img[roi[0]:roi[1],roi[2]:roi[3]]
    img_mask_R=np.abs(R>np.clip(B.astype(np.int)+40,0,255)).astype(np.uint8)
    y_sum_R=np.sum(img_mask_R,axis=0)
    y_sum_R=y_sum_R>60
    img_roi=cv2.transpose(img_roi)
    if np.sum(y_sum_R[:300])<=np.sum(y_sum_R[300:]):
        img_roi=cv2.flip(img_roi,0)
    B,G,R=cv2.split(img_roi)
    img_mask_G=np.abs((G>np.clip(B.astype(np.int)+40,0,255))*G).astype(np.uint8)
    img_mask_R=(R>np.clip(B.astype(np.int)+40,0,255))
    circles_Y=cv2.HoughCircles(img_mask_G, cv2.HOUGH_GRADIENT, 1, 10,
                        param1=2, param2=10, minRadius=5, maxRadius=50)
    print(circles_Y)
    x,y,r=int(circles_Y[0][0][0]),int(circles_Y[0][0][1]),int(circles_Y[0][0][2])
    img_roi=img_roi[y+r:y+100,x-50:x+150]
    img_mask_R=img_mask_R[y+r:y+100,x-50:x+150]
    img_roi_gray=cv2.cvtColor(img_roi,cv2.COLOR_BGR2GRAY)
    img_roi_gray=cv2.medianBlur(img_roi_gray,5)
    #img_roi_gray=cv2.equalizeHist(img_roi_gray)
    ret,thresh1 = cv2.threshold(img_roi_gray,0,255,cv2.THRESH_OTSU)
    img_roi_bin=~img_mask_R*thresh1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_roi_bin= cv2.morphologyEx(img_roi_bin, cv2.MORPH_CLOSE, kernel, iterations=5)
    img_roi_bin=cv2.GaussianBlur(img_roi_bin,(3,3),0)
    img_roi_edge=cv2.Canny(img_roi_bin,1,2)
    lines = cv2.HoughLinesP(img_roi_edge, 1, np.pi/180, 20, minLineLength=30, maxLineGap=30)
    point_color = (0, 255, 0) # BGR
    thickness = 1
    lineType = 4
    avg_degree=0
    cnt=0
    for line in lines:
        line=line[0]
        degree=np.math.atan2(line[3]-line[1],line[2]-line[0])/np.math.pi*180
        if abs(degree):
            avg_degree+=degree
            cnt+=1
            dist=point_distance_line(np.array([x,y]),np.array(line[:2]),np.array(line[2:]))
            print(dist)
        cv2.line(img_roi,(line[0],line[1]),(line[2],line[3]), point_color, thickness, lineType)
    avg_degree/=cnt
    x+=roi[2]
    y+=roi[0]
    M = cv2.getRotationMatrix2D((x,y), degree, 1.0)
    img = cv2.warpAffine(img, M, (width, height))
    return img
img_out=cropper(img)
plt.imshow(img_out,cmap="gray")
# %%
y_sum_R=np.sum(img_mask_R,axis=0)
y_sum_R=y_sum_R>60
if np.sum(y_sum_R[:300])>np.sum(y_sum_R[300:]):
    direction=0
else:
    direction=1
print(direction)
plt.plot(y_sum_R)
# %%
plt.plot(y_sum_R[1:]^y_sum_R[:-1])

# %%
np.where(y_sum_R[1:]^y_sum_R[:-1])
# %%
img=img[150:300,:]
plt.imshow(img)
# %%
dst = cv2.Canny(img_mask_R, 1,2, None, 3)
plt.imshow(dst,cmap="gray")
#%%
lines = cv2.HoughLinesP(dst, 1, np.pi/180, 50, minLineLength=50, maxLineGap=5)
if lines is not None:
    for i in range(0, len(lines)):
        l = lines[i][0]
        cv2.line(dst, (l[0], l[1]), (l[2], l[3]), (255), 3, cv2.LINE_AA)
print(lines)
plt.imshow(dst,cmap="gray")
# %%
plt.imshow(img)
# %%
dst = cv2.Canny(img_mask_R, 1,2, None, 3)
plt.imshow(dst)
# %%
vertical=img_mask_R[185:195,:]
plt.imshow(vertical,cmap="gray")
# %%
vertical_sum=np.sum(vertical,axis=0)
plt.plot(vertical_sum)
# %%
import json
with open("Panel.json","r") as f:
    obj=json.load(f)
# %%
print(obj.keys())

# %%
print(obj["_via_img_metadata"].keys())
# %%
import cv2
import numpy as np
for key in obj["_via_img_metadata"].keys():
    file_name=obj["_via_img_metadata"][key]["filename"]
    regions=obj["_via_img_metadata"][key]["regions"]

    biggest_area=0
    for region in regions:
        rect=region['shape_attributes']
        area=rect['width']*rect['height']
        if area>biggest_area:
            pt1=(rect['x'],rect['y'])
            pt2=(rect['x']+rect['width'],rect['y']+rect['height'])
            biggest_area=area
    img=cv2.imread(os.path.join("output_1",file_name))
    img_mask=np.zeros((img.shape[0],img.shape[1],1),dtype=np.uint8)
    img_mask[pt1[1]:pt2[1],pt1[0]:pt2[0],:]=1
    img=img_mask*img
    cv2.imwrite(file_name,img)
# %%
from matplotlib import pyplot as plt
img=cv2.imread("panel_mask/C.jpg")
detector=cv2.SIFT_create(100)
ref_kp=detector.detect(img,None)
ref_kp,ref_des=detector.compute(img,ref_kp)
cv2.drawKeypoints(img,ref_kp,img,color=(0,255,0))
plt.imshow(img)

# %%
target_img=cv2.imread("output_1/WIN_20210123_13_43_50_Pro.jpg")
bf=cv2.BFMatcher_create()
target_kp=detector.detect(target_img,None)
target_kp,target_des=detector.compute(target_img,target_kp)
matches=bf.knnMatch(ref_des,target_des,k=2)
matchesMask = [[0,0] for i in range(len(matches))]
match_points=[]
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        match_points.append(m)
#将特征点的匹配关系进行绘制
draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = 0)
ref_pts = np.float32([ ref_kp[m.queryIdx].pt for m in match_points ]).reshape(-1,1,2)
target_pts = np.float32([ target_kp[m.trainIdx].pt for m in match_points ]).reshape(-1,1,2)
H, mask = cv2.findHomography(target_pts,ref_pts, cv2.RANSAC,10.0,confidence=0.8)
target_img=cv2.warpPerspective(target_img,H,dsize=(target_img.shape[1],target_img.shape[0]))
plt.imshow(target_img)
# %%
from matplotlib import pyplot as plt
import cv2
import numpy as np
plt.imshow(img)
# %%
img=cv2.imread("B/0088.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
b_hist = plt.hist(img.ravel(), bins=50, color='b',range=(0,255))
# %%
thresh,bin_img=cv2.threshold(img,20,255,cv2.THRESH_BINARY)
plt.imshow(bin_img,cmap="gray")
# %%

cv2.imwrite("test.jpg",bin_img)
# %%
def findsplitline(hist):
    #找出所有非0数据的位置
    width=[]
    start=0
    nonzero=np.nonzero(hist)[0]
    #找出非0数据的开始与结束
    if len(nonzero)>0:
        start=nonzero[0]-5 if nonzero[0]>5 else 0
        left_border=nonzero[1:]
        right_border=nonzero[:-1]
        border=np.nonzero(left_border-right_border-1)[0]
        left_border=left_border[border]
        right_border=right_border[border]
        lines=(left_border+right_border)//2
        
        if len(lines)==12:
            split=lines[1:]-lines[:-1]
            meanwidth=round(np.mean(split))
            width.append(lines[0]-start)
            width.extend(split.tolist())
            width.append(meanwidth)
    return start,width

start,width=findsplitline(np.sum(img[20:90,10:],axis=0))
# %%
thresh,img=cv2.threshold(img,40,255,cv2.THRESH_OTSU)

plt.plot(np.sum(255-img[20:90,10:],axis=0))
# %%
hist
# %%
from glob import glob
import os
img_path_list=glob("data/input/A/*.jpg")
idx=0
for img_path in img_path_list:
    img=cv2.imread(img_path)
    img=img[:1200,:1600]
    img=cv2.resize(img,(640,480))
    cv2.imwrite(f"data/input/C/{idx}.jpg",img)
    idx+=1
# %%
import cv2
from matplotlib import pyplot as plt
img=cv2.imread("panel_mask/A.jpg")
# %%
plt.imshow(img)
# %%
M = cv2.getRotationMatrix2D((100,300), -0, 1.0)
rotate_img = cv2.warpAffine(img, M, (480, 640))
plt.imshow(rotate_img)
# %%
cropped_img=rotate_img[155:302,155:323]
plt.imshow(cropped_img)
# %%
rotate_img[550:600,200:400]=[0,0,0]
plt.imshow(rotate_img)
#%%
img=cv2.imread("panel_mask/B_2.jpg")
circle_img=cv2.circle(img,(400,340),59,[0,0,0],-1)
circle_img[79:195,312:480]=[0,0,0]
plt.imshow(circle_img)
#%%
cv2.imwrite("panel_mask/C.jpg",circle_img)
# %%
from glob import glob
import numpy as np
img_path_list=sorted(glob("data/output/C/*.jpg"))
# %%
for img_path in img_path_list:
    img=cv2.imread(img_path)
    img_roi=img[:,:]
    #img_roi=cv2.cvtColor(img_roi,cv2.COLOR_BGR2GRAY)
    img_roi=cv2.blur(img,(3,3))
    img_roi=cv2.Sobel(img,cv2.CV_64F,0,1)

    img_roi=cv2.convertScaleAbs(img_roi)
    img_roi=cv2.cvtColor(img_roi,cv2.COLOR_BGR2GRAY)
    thresh,img_roi=cv2.threshold(img_roi,128,255,cv2.THRESH_BINARY)
    lines=cv2.HoughLinesP(img_roi,1,np.math.pi/180,50,minLineLength=20, maxLineGap=5)
    min_line=130
    max_line=5
    for line in lines:
        line=line[0]
        if line[1]==line[3]:
            if line[1]<20 and line[1]>max_line:
                max_line=line[1]
            if line[1]>120 and line[1]<min_line:
                min_line=line[1]
    print(max_line,max_line-min_line)
# %%
from glob import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

img_path_list=sorted(glob("data/output/C/*.jpg"))
for img_path in img_path_list:
    img=cv2.imread(img_path)
    img_roi=img[30:110,20:175]
    img_roi_gray=cv2.cvtColor(img_roi,cv2.COLOR_BGR2GRAY)
    thresh,img_roi_bin=cv2.threshold(img_roi_gray,0,255,cv2.THRESH_OTSU)
    print(thresh)
    y_sum=np.sum(img_roi_bin<128,1)
    print(y_sum)
    none_zero=np.nonzero(y_sum>10)[0]
    print(none_zero[0],none_zero[-1],none_zero[-1]-none_zero[0])
    img_roi=cv2.line(img,(0,30+none_zero[0]),(180,30+none_zero[0]),[0,255,255],2)
    img_roi=cv2.line(img,(0,30+none_zero[-1]),(180,30+none_zero[-1]),[0,255,255],2)

    cv2.imwrite(f"data/output/test/{os.path.basename(img_path)}",img_roi)

# %%
model=cv2.dnn_TextDetectionModel_EAST()
# %%
img=cv2.imread("panel_mask/B_2.jpg")
plt.imshow(img)
# %%
M = cv2.getRotationMatrix2D((100,300), -1.5, 1.0)
rotate_img = cv2.warpAffine(img, M, (480, 640))
img_roi=rotate_img[305:352,185:280]
plt.imshow(img_roi)
# %%
rotate_img[305:352,185:280]=[0,0,0]
cv2.imwrite("panel_mask/B.jpg",rotate_img)
# %%
import numpy as np

a=np.arange(10)
print(a[3:9:2])
# %%
