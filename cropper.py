import json
import pickle
import cv2
import numpy as np
import os
import shutil

class Cropper():
    def __init__(self, cfg) -> None:
        self.detector = cv2.AKAZE_create()
        self.bf = cv2.BFMatcher_create()
        self.ref_infos=[]
        for key in cfg["masks"].keys():
            for info in cfg["masks"][key]:
                ref_img = cv2.imread(info["img_path"])
                with open(info["mask_path"], "rb") as f:
                    mask_info = pickle.load(f)
                    mask, roi = mask_info["mask"], mask_info["roi"]
                ref_kp = self.detector.detect(ref_img*mask[:,:,None], mask)
                ref_kp, ref_des = self.detector.compute(ref_img, ref_kp)
                height, width = ref_img.shape[0], ref_img.shape[1]
                roi = (slice(roi[1], roi[1]+roi[3]),
                       slice(roi[0], roi[0]+roi[2]))
                self.ref_infos.append(
                    (key, ref_img, ref_kp, ref_des, (width, height), mask, roi))
        self.message_list=["Success","Not enough feature point detected","System Error","Few feature point detected,please add a mask"]

    def classify(self, target_des):
        best_match, best_match_idx = [], 0
        for idx, info in enumerate(self.ref_infos):
            match_points = self.get_matches(info[3],target_des)
            if len(match_points) > len(best_match):
                best_match, best_match_idx = match_points, idx
        return best_match_idx, best_match
    def get_matches(self,ref_des,target_des):
        matches = self.bf.knnMatch(ref_des, target_des, k=2)
        return [m for m, n in matches if m.distance < 0.8*n.distance]
    def dist_filter(self, target_pts, ref_pts):
        H, mask = cv2.findHomography(
            target_pts, ref_pts, cv2.RANSAC, 10.0, confidence=0.7)
        dist_filter=np.zeros(target_pts.shape[0],dtype=np.bool8)
        if type(H) is np.ndarray:
            cols = np.ones((target_pts.shape[0], 3, 1))
            cols[:, 0:2, :] = target_pts.reshape((-1, 2, 1))
            cols = np.dot(H, cols)
            cols /= cols[2, :, :]
            dist_filter = np.sqrt(
                (cols[0]-ref_pts[:, :, 0])**2+(cols[1]-ref_pts[:, :, 1])**2) < 1
        return dist_filter
    def post_process(self,img):
        img = cv2.transpose(img)
        img = cv2.flip(img, 0)
        # img = cv2.fastNlMeansDenoisingColored(
        #             img, None, 5, 5, 7, 31)
        return img
    def __call__(self, target_img):
        try:
            target_kp = self.detector.detect(target_img, None)
            target_kp, target_des = self.detector.compute(
                target_img, target_kp)
            status=0
            idx, match_points = self.classify(target_des)
            if len(match_points) > 10:
                key,ref_kp, ref_des, dsize,ref_mask,roi,= self.ref_infos[idx][0], self.ref_infos[idx][2], self.ref_infos[idx][3], self.ref_infos[idx][4], self.ref_infos[idx][5], self.ref_infos[idx][6]
                for i in range(2):
                    if i > 0:
                        target_kp = self.detector.detect(target_img,ref_mask)
                        target_kp, target_des = self.detector.compute(
                            target_img, target_kp)
                        match_points=self.get_matches(ref_des,target_des)
                    ref_pts = np.float32(
                        [ref_kp[m.queryIdx].pt for m in match_points]).reshape(-1, 1, 2)
                    target_pts = np.float32(
                        [target_kp[m.trainIdx].pt for m in match_points]).reshape(-1, 1, 2)

                    dist_filter = self.dist_filter(target_pts, ref_pts)
                    if np.sum(dist_filter) > 10:
                        ref_pts = ref_pts[dist_filter]
                        target_pts = target_pts[dist_filter]

                        H, mask = cv2.findHomography(
                            target_pts, ref_pts, cv2.RHO, 5.0, confidence=0.95, maxIters=100)
                        target_img = cv2.warpPerspective(
                            target_img, H, dsize=dsize)
                        if np.sum(dist_filter)<75:
                            status=3
                        else:
                            status=0
                    else:
                        status=1
                        break

                target_img = target_img[roi[0], roi[1]]
                target_img=self.post_process(target_img)
            else:
                status=1
        except:
            status=2
        if status==0:
            return status,target_img,key,self.message_list[status]
        else:
            return status,None,-1,self.message_list[status]



if __name__ == "__main__":
    from camera import Camera
    output_dir = "data/output/B"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    with open("config.json","r") as f:
        cfg=json.load(f)
        cropper_cfg=cfg["cropper"]
        camera_cfg=cfg["camera"]
    camera=Camera(camera_cfg)
    cropper=Cropper(cropper_cfg)
    idx=0
    while(1):
        ret,frame=camera()
        if ret:
            status,panel_img,key,message=cropper(frame)
            cv2.imshow("img",frame)
            if status==0:
                cv2.imshow("panel",panel_img)
            else:
                print(message)
        else:
            break
        k=cv2.waitKey(1)
        if k== ord('q'):
            break
        elif k==ord("s"):
            output_path=os.path.join(output_dir,key,f"{idx}.jpg")
            if not os.path.exists(os.path.join(output_dir,key)):
                os.makedirs(os.path.join(output_dir,key))
            cv2.imwrite(output_path,panel_img)
            print(f"image saved at {output_path}")
            idx+=1
    camera.release()
    cv2.destroyAllWindows()
