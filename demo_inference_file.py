# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""
import base64
import io
import os
import sys
import numpy as np
import logging
import argparse
from pathlib import Path
import cv2
from paddleocr import PaddleOCR, draw_ocr

import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import matplotlib.pyplot as plt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolo.models.common import DetectMultiBackend
from yolo.utils.datasets1 import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolo.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolo.utils.plots import Annotator, colors, save_one_box
from yolo.utils.torch_utils import select_device, time_sync

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
_logger = logging.getLogger(__name__)


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'yolo/data/images',  # file/dir/URL/glob, 0 for webcam
        # source=bytes,
        data=ROOT / 'yolo/data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    data_list = []
    source = str(source)
    # source = base64.b64decode(source)
    # device = select_device(device)
    device = torch.device('cuda:0' if False else 'cpu')
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # dataset = Image.open(io.BytesIO(source))
    # dataset = dataset.load()
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if float(conf) > 0.6:
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        class_val = [label, int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        data_list.append(class_val)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    return data_list


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    # parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str)
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def region_list_accuracy(data_list, class_info):
    """ List of Regions Accuracy """
    region_list = []

    for idx, i_data in enumerate(class_info):
        count_val = [i for i in data_list if i_data in i[0]]
        if count_val:
            if len(count_val) > 1:
                sort_acc = count_val[np.argmax([float(p_data[0].split(' ')[1]) for p_data in count_val if p_data[0]])]
                region_list.append(sort_acc)
            else:
                region_list.append(count_val[0])

    return region_list


def region_detection(source_img, model_file_name):
    """ Region Detection """
    region_list, class_info = [], []
    try:
        data_list = run(model_file_name, source=source_img)
        region_list = region_list_accuracy(data_list, ['car_lisence_plate'])
        check_region = check_region_detection(source_img, region_list)
        region_plots = cut_the_image_with_region(source_img, region_list)
    except Exception as e:
        _logger.error("Region Detection Process is failed ! " + repr(e))

    return region_list


def check_region_detection(source_image, predicted_region_list):
    try:
        image_bytes = base64.b64decode(source_image)
        image_array = np.frombuffer(image_bytes, np.uint8)
        # Load the image using OpenCV
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        for coordinates in predicted_region_list:
            color = (0, 255, 0)  # Green color (BGR format)
            thickness = 2
            font_scale = 0.6
            font_thickness = 2
            text_color = (0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            # if coordinates[0][0:3] in write_text_only_values:
            #     text_x = coordinates[1]
            #     text_y = coordinates[2]  # Adjust this value to change the vertical position of the label
            #     cv2.putText(image, coordinates[0], (text_x, text_y), font, font_scale, text_color, font_thickness)
            # else:
            x1, y1 = coordinates[1], coordinates[2]  # Top-left corner
            x2, y2 = coordinates[3], coordinates[4]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            text_x = x1
            text_y = y1 - 10  # Adjust this value to change the vertical position of the label
            cv2.putText(image, coordinates[0], (text_x, text_y), font, font_scale, text_color, font_thickness)
        cv2.imwrite('output_image.jpg', image)
        return True
    except Exception as e:
        print(f'image saving process has been failed, because of {e}')
        return False


def cut_the_image_with_region(image_string, region_plots):
    try:
        nparr = np.fromstring(base64.b64decode(image_string), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print('image read successfully')
        roi_x, roi_y, roi_width, roi_height = region_plots[0][1], region_plots[0][2], region_plots[0][3], region_plots[0][4]
        print("ROI coordinates defined.")
        crop_image = cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
        roi = img[roi_y:roi_height, roi_x:roi_width]
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        ocr = PaddleOCR(lang='en')
        result = ocr.ocr(roi_rgb, cls=True)
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                print(f'The Vehicle number is: {line[1][0]}')
        # pil_img = Image.fromarray(roi_rgb)
        # buff = io.BytesIO()
        # pil_img.save(buff, format="JPEG")
        # new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
        return img
    except Exception as e:
        print(e)

