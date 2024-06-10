import os
import shutil
from bs4 import BeautifulSoup


def normalized_coordinates(filename, width, height, xmin, ymin, xmax, ymax):
    """Take in image coordinates (unnormalized) as input, return normalized values
    """

    xmin, xmax = xmin / width, xmax / width
    ymin, ymax = ymin / height, ymax / height

    width = xmax - xmin
    height = ymax - ymin
    x_center = xmin + (width / 2)
    y_center = ymin + (height / 2)

    return x_center, y_center, width, height


def write_label(filename, x_center, y_center, width, height):
    """Save image's coordinates in text file named "filename"
    """
    with open(filename, mode='w') as outf:
        outf.write(f"{0} {x_center} {y_center} {width} {height}\n")


def parse_xml_tags(data):
    """Parse xml label file, return image file name, and its coordinates as a dictionary
    """
    tags = ['filename', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']
    Bs_data = BeautifulSoup(data, "xml")
    d = dict()

    for t in tags:
        text = Bs_data.find(t).text
        if all(c.isdigit() for c in text):
            d[t] = int(text)
        else:
            d[t] = text
    return d


def build_data(dir_folder, ann_file_list, img_dir):
    """Write xml labels to text file with specifications format, save at 'labels' folder.
        Move image to 'images' folder
    """
    images_folder = f"{dir_folder}/images"
    labels_folder = f"{dir_folder}/labels"

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    for ann_file in os.listdir(ann_file_list):
        with open(os.path.join(ann_file_list, ann_file), 'r') as f:
            label = parse_xml_tags(f.read())

        img_file_name = label['filename']
        x_center, y_center, width, height = normalized_coordinates(**label)

        # save at 'labels' folder
        write_label(f"{labels_folder}/{img_file_name.split('.')[0]}.txt", x_center, y_center, width, height)

        # Move image to 'images' folder
        shutil.copy(f"{img_dir}/{img_file_name}", f"{images_folder}/{img_file_name}")


dir_folder = r'C:\Users\VR DELLA\Desktop\Number_plate_detection\number_plate_dataset'
annotation_folder = r'C:\Users\VR DELLA\Downloads\car_number_plate_dataset\annotations'
image_folder = r'C:\Users\VR DELLA\Downloads\car_number_plate_dataset\images'
build_data(dir_folder, annotation_folder, image_folder)
