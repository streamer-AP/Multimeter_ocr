import cv2
import os
import numpy as np
from functools import partial
from camera import Camera
import json
import pickle

cfg={
    "x1":0,"y1":0,"x2":0,"y2":0,"x3":0,"y3":0,"w1":0,"h1":0,"w2":0,"h2":0,"r3":0
}
def change_config_wrapper(key):
    def change_config(value,key):
        cfg[key]=value
    return partial(change_config,key=key)

if __name__ == '__main__':
    config_file_name="config.json"
    output_dir="data/masks/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(config_file_name,"r") as f:
        camera_cfg=json.load(f)["camera"]
    camera=Camera(camera_cfg)
    cv2.namedWindow("img")
    for key in cfg.keys():
        cv2.createTrackbar(key,'img',0,1000,change_config_wrapper(key))
    detector=cv2.AKAZE_create()
    idx=0
    print(cfg)
    dst=cv2.imread("data/masks/C.png")
    while(1):

        #dst = cv2.fastNlMeansDenoisingColored(frame,None,3,3,7,21)
        mask=np.zeros_like(dst[:,:,0])
        cv2.rectangle(mask,(cfg["x1"],cfg["y1"]),(cfg["x1"]+cfg["w1"],cfg["y1"]+cfg["h1"]),[1],-1)

        cv2.rectangle(mask,(cfg["x2"],cfg["y2"]),(cfg["x2"]+cfg["w2"],cfg["y2"]+cfg["h2"]),[0],-1)
        cv2.circle(mask,(cfg["x3"],cfg["y3"]),cfg["r3"],[0],-1)

        target_kp=detector.detect(dst,mask)
        target_kp,target_des=detector.compute(dst,target_kp)
        dst_show=dst.copy()
        cv2.rectangle(dst_show,(cfg["x1"],cfg["y1"]),(cfg["x1"]+cfg["w1"],cfg["y1"]+cfg["h1"]),[0,255,255],1)
        cv2.rectangle(dst_show,(cfg["x2"],cfg["y2"]),(cfg["x2"]+cfg["w2"],cfg["y2"]+cfg["h2"]),[0,255,255],1)
        cv2.circle(dst_show,(cfg["x3"],cfg["y3"]),cfg["r3"],[0,255,255],1)
        
        dst_show = cv2.drawKeypoints(dst_show, target_kp, None)
        cv2.imshow("img",dst_show)

        k=cv2.waitKey(1)
        if k== ord('q'):
            break
        elif k==ord("s"):
            with open(os.path.join(output_dir,f"C.pkl"),"wb") as f:
                pickle.dump({"mask":mask,"roi":[cfg["x2"],cfg["y2"],cfg["w2"],cfg["h2"]],"knob":[cfg["x3"],cfg["y3"],cfg["r3"]]},f)
            cv2.imwrite(os.path.join(output_dir,f"C.png"),dst)
            idx+=1
            print(idx)
    camera.release()
    cv2.destroyAllWindows() 
