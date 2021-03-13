# coding:utf-8
import json
import time

import cv2
import torch

from torchvision.models import resnet18
from torchvision.transforms import Compose, Resize, ToTensor

from camera import Camera
from cropper import Cropper
from utils import get_digit_imgs,get_digits,get_unit  
import base64
import time
with open("config.json", "r") as f:
    cfg = json.load(f)
    cropper_cfg = cfg["cropper"]
    camera_cfg = cfg["camera"]
    reco_cfg = cfg["reco"]
    unit_cfg = cfg["unit"]
    server_cfg = cfg["server"]

    camera = Camera(camera_cfg)
    cropper = Cropper(cropper_cfg)

    if reco_cfg["model_name"] == "resnet18":
        reco_model = resnet18(num_classes=reco_cfg["num_classes"])
    reco_model.load_state_dict(torch.load(
        reco_cfg["model_path"])["model_state_dict"])
    reco_model.eval()

    reco_char_dict = reco_cfg["char_dict"]
    transform = ToTensor()

    if unit_cfg["model_name"] == "resnet18":
        unit_model = resnet18(num_classes=unit_cfg["num_classes"])
    unit_model.eval()
    unit_model.load_state_dict(torch.load(unit_cfg["model_path"])["model_state_dict"])
    unit_char_dict = unit_cfg["char_dict"]
    unit_transforms = Compose([Resize((unit_cfg["height"], unit_cfg["width"])), ToTensor()])
    while True:
        ret, frame = camera()
        if ret:
            cv2.imwrite("tmp.jpg",frame)
            start=time.time()
            print(ret)
            digits=[],
            unit="",
            if ret:
                status, panel_img,knob_img, key, message = cropper(frame)
                digits = []
                if status == 0 or status == 3:
                    digit_imgs= get_digit_imgs(
                        panel_img,reco_cfg[key]["top_ratio"],reco_cfg[key]["bottom_ratio"],reco_cfg[key]["pre_bbox"],reco_cfg[key]["pre_char_num"],reco_cfg[key]["pre_char_width"],reco_cfg[key]["pre_first_char_offset"])
                    digits=get_digits(digit_imgs, reco_model, reco_char_dict)
                    unit=get_unit(knob_img,unit_model,unit_transforms
                    ,unit_char_dict)
                print(digits)
                print(unit)
            else:
                message="Camera error"
            print(time.time()-start)
            
            img=base64.b64encode(frame.tostring()).decode("ascii")
            result=json.dumps({"digit":digits,"unit":unit,"message":message,"img":img})
            print(message)
            cv2.imshow("frame",frame)
            cv2.waitKey(0)
        else:
            break