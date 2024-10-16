import json
import os


def coco_to_yolo(coco_json_file, output_txt_file, class_names):
    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)

    width = coco_data["imageWidth"]
    height = coco_data["imageHeight"]

    with open(output_txt_file, 'w') as f:
        for shape in coco_data['shapes']:
            label = shape['label']
            # class_label = class_names.index(label) + 1
            class_label = class_names.index(label)

            points = shape['points']
            x_min = min(points[0][0], points[1][0])
            y_min = min(points[0][1], points[1][1])
            x_max = max(points[0][0], points[1][0])
            y_max = max(points[0][1], points[1][1])

            # Convert bounding box coordinates to YOLO format
            x_center = (x_min + x_max) / (2 * width)
            y_center = (y_min + y_max) / (2 * height)
            box_width = (x_max - x_min) / width
            box_height = (y_max - y_min) / height

            # Write annotation in YOLO format to the output file
            f.write(f"{class_label} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")


# Input directory containing COCO JSON files
input_dir = r"C:\Users\VR DELLA\Downloads\lead_aVL\Lead aVL\Annotated_json"

# Output directory where YOLO format text files will be saved
output_dir = r'C:\Users\VR DELLA\Downloads\lead_aVL\Lead aVL\Annotated_txt'

# Class names (modify this according to your dataset)
# class_names = ["S_Valley", "T_Wave", "P_Wave", "Q_Valley"]
# class_names = ['car_lisence_plate']
class_names = ["P_Wave", "T_Wave", "QRS_Complex"]

# Iterate over each JSON file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        print("##########filename", filename)
        json_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename.replace('.json', '.txt'))
        coco_to_yolo(json_file, output_file, class_names)
