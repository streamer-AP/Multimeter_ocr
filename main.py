import json
import os
import shutil

import cv2

from camera import Camera
from cropper import Cropper
from Reco_A import scan as scan_a
from Reco_B import scan as scan_b
from Reco_C import scan as scan_c

scanner = {
    "A": scan_a, "B": scan_b, "C": scan_c
}
if __name__ == "__main__":
    output_dir = "data/output/B"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    with open("config.json", "r") as f:
        cfg = json.load(f)
        cropper_cfg = cfg["cropper"]
        camera_cfg = cfg["camera"]
    camera = Camera(camera_cfg)
    cropper = Cropper(cropper_cfg)
    idx = 0
    while(1):
        ret, frame = camera()
        if ret:
            status, panel_img, key, message = cropper(frame)
            cv2.imshow("img", frame)
            if status == 0:
                cv2.imshow("panel", panel_img)
                digits, debug_img = scanner[key](panel_img)
                cv2.imshow("reco", debug_img)
                print(digits)
            else:
                print(message)
        else:
            break
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord("s"):
            output_path = os.path.join(output_dir, key, f"{idx}.jpg")
            if not os.path.exists(os.path.join(output_dir, key)):
                os.makedirs(os.path.join(output_dir, key))
            cv2.imwrite(output_path, panel_img)
            print(f"image saved at {output_path}")
            idx += 1
    camera.release()
    cv2.destroyAllWindows()
