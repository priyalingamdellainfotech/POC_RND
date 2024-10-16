import json
import os

json_dir = r'C:\Users\VR DELLA\Downloads\Lead2_json_files 1\Lead2_json_files'

if os.path.exists(json_dir) and os.listdir(json_dir):
    for annotate_file in os.listdir(json_dir):
        with open(os.path.join(json_dir, annotate_file), 'r') as file:
            coco_file = json.load(file)
        print('file opened')
        for shapes in coco_file['shapes']:
            print(shapes)
            if shapes['points'][0][0] > shapes['points'][1][0]:
                shapes['points'][0][0], shapes['points'][1][0] = shapes['points'][1][0], shapes['points'][0][0]
            if shapes['points'][0][1] > shapes['points'][1][1]:
                shapes['points'][0][1], shapes['points'][1][1] = shapes['points'][1][1], shapes['points'][0][1]
            print('changed')
        with open(os.path.join(json_dir, annotate_file), 'w') as f:
            json.dump(coco_file, f)
