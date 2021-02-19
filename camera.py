import cv2
import numpy as np


class Camera():
    def __init__(self, camera_cfg) -> None:
        width, height, fps, calibration, origin_roi, calibrated_roi = camera_cfg["width"], camera_cfg[
            "height"], camera_cfg["fps"], camera_cfg["params"], camera_cfg["origin_roi"], camera_cfg["calibrated_roi"]
        self.cap = cv2.VideoCapture(camera_cfg["file_path"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))
        self.origin_roi = origin_roi
        self.calibrated_roi = calibrated_roi
        calibration = np.load("calibration.npy", allow_pickle=True)
        mtx = calibration[()]["mtx"]
        dist = calibration[()]["dist"]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (origin_roi[2], origin_roi[3]), 1, (origin_roi[2], origin_roi[3]))
        self.mtx, self.dist, self.camerametx = mtx, dist, newcameramtx

    def __call__(self,):
        ret, frame = self.cap.read()
        if ret == 1:
            frame = frame[self.origin_roi[1]:self.origin_roi[1]+self.origin_roi[3],
                          self.origin_roi[0]:self.origin_roi[0]+self.origin_roi[2]]
            frame = cv2.undistort(
                frame, self.mtx, self.dist, None, self.camerametx)
            frame = frame[self.calibrated_roi[1]:self.calibrated_roi[1]+self.calibrated_roi[3],
                          self.calibrated_roi[0]:self.calibrated_roi[0]+self.calibrated_roi[2]]
        return ret, frame

    def release(self):
        self.cap.release()


if __name__ == '__main__':
    import json
    import os

    config_file_name = "config.json"
    output_dir = "data/output/test/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(config_file_name, "r") as f:
        cfg = json.load(f)
    camera_cfg = cfg["camera"]
    camera = Camera(camera_cfg)
    video_writer = cv2.VideoWriter('record_C.avi', cv2.VideoWriter_fourcc(*'MJPG'),camera_cfg["fps"],  (camera_cfg["width"], camera_cfg["height"]), )
    idx = 0
    while(1):
        ret, frame = camera.cap.read()
        if ret:
            cv2.imshow("img", frame)
            video_writer.write(frame)
        else:
            break
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord("s"):
            output_path = os.path.join(output_dir, f"{idx}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"image saved at {output_path}")

            idx += 1
    camera.release()
    video_writer.release()
    cv2.destroyAllWindows()
