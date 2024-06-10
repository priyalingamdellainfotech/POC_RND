import os
import base64
import argparse
from demo_inference_file import region_detection


def model_run_operation(source_image, weights):
    try:
        with open(source_image, 'rb') as file:
            source_image = base64.b64encode(file.read()).decode('utf-8')
        # source_image = base64.b64decode(source_image)
        data = region_detection(source_image, weights)
        return data
    except Exception as e:
        print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_image", type=str, default="C:/Users/VR DELLA/Downloads/car.jpg", help='place the predict image source dir / file path')
    parser.add_argument("--weights", type=str, default="C:/Users/VR DELLA/Downloads/best-fp16.tflite", help="send the model file path")
    opt = parser.parse_args()
    model_run_operation(**vars(opt))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
