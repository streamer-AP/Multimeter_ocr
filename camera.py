import cv2
import numpy as np
from cropper import Cropper
import threading
import queue

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

        calibration = np.load(calibration, allow_pickle=True)
        mtx = calibration[()]["mtx"]
        dist = calibration[()]["dist"]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (origin_roi[2], origin_roi[3]), 1, (origin_roi[2], origin_roi[3]))
        self.mtx, self.dist, self.camerametx = mtx, dist, newcameramtx
        self.q=queue.Queue()
        # t=threading.Thread(target=self._reader)
        # t.daemon=True
        # t.start()

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

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if ret == 1:
                if not self.q.empty():
                    try:
                        self.q.get_nowait()
                    except:
                        pass
                self.q.put((ret,frame))
            else:
                break
    def release(self):
        self.cap.release()