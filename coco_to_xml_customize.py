import json
import os
import argparse
import logging
import pandas as pd

_logger = logging.getLogger(__name__)


def write_to_xml(image_name, bboxes, image_folder_name, coco_data, save_folder, database_name=''):
    depth = ''
    objects = ''
    print(f'XML converting process started for {coco_data["imagePath"]}....')
    for bbox in bboxes:
        objects = objects + '''
        <object>
            <name>{category_name}</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>{xmin}</xmin>
                <ymin>{ymin}</ymin>
                <xmax>{xmax}</xmax>
                <ymax>{ymax}</ymax>
            </bndbox>
        </object>'''.format(
                category_name=bbox['label'],
                xmin=bbox['points'][0][0],
                ymin=bbox['points'][0][1],
                xmax=bbox['points'][1][0],
                ymax=bbox['points'][1][1]
            )

        xml = '''<annotation>
        <folder>{image_folder_name}</folder>
        <filename>{image_name}</filename>
        <source>
            <database>{database_name}</database>
        </source>
        <size>
            <width>{width}</width>
            <height>{height}</height>
            <depth>{depth}</depth>
        </size>
        <segmented>0</segmented>{objects}
    </annotation>'''.format(
            image_folder_name=image_folder_name,
            image_name=image_name,
            database_name=database_name,
            width=coco_data['imageWidth'],
            height=coco_data['imageHeight'],
            depth=depth,
            objects=objects
        )

        anno_path = os.path.join(save_folder, os.path.splitext(image_name)[0] + '.xml')

        with open(anno_path, 'w') as file:
            file.write(xml)
    print(f'Hurrah!!... {os.path.splitext(image_name)[0] + ".xml"} file stored in given path....')


parser = argparse.ArgumentParser(description='This method is for convert the json annotation file to xml file for rcnn')

parser.add_argument('--json_data', required=True, help='This folder of json data which is going to convert')
parser.add_argument('--save_location', required=True, help='The folder name and path which is converted file going to store')
parser.set_defaults(skip_background=True)

args = parser.parse_args()

unmatched_file = []
try:
    if os.path.exists(args.json_data) and os.listdir(args.json_data):
        print('path exists... good to go....!!')
        if not os.path.exists(args.save_location):
            os.makedirs(args.save_location)
        list_dir = os.listdir(args.json_data)
        for idx, annotation_file in enumerate(list_dir):
            if annotation_file.endswith('.json'):
                print(f'Processing.... {annotation_file}')
                file_name = os.path.join(args.json_data, annotation_file)
                with open(file_name, 'r') as f:
                    coco_data = json.load(f)
                print(f'******* {annotation_file}, file read done!! *******')
                write_to_xml(coco_data['imagePath'], coco_data['shapes'], args.json_data.split('/')[-1], coco_data, args.save_location)
                print('Process Almost done!!', float(idx) / len(list_dir), '% ~~~')
            else:
                unmatched_file.append(annotation_file)
        if unmatched_file:
            df = pd.DataFrame()
            df['Sno'] = [i for i in range(1, len(unmatched_file))]
            df['unsaved_files'] = unmatched_file
            df.to_excel('unsaved_files_list.xlsx', index=False)
            print('Unfortunately!!, Some files are not created... You can see that list in unsaved_files_list.xlsx file in this path!!...')

    else:
        raise Exception(f'{args.json_path} json path not exists or that is an empty directory....')

except Exception as e:
    _logger.error(str(e))

