import json
import os
import shutil
import argparse
import logging
import pandas as pd

_logger = logging.getLogger(__name__)


def image_and_json_file_validation(data_dir, destination_path):
    try:
        files = os.listdir(data_dir)
        image_ext = '.png'
        json_ext = '.json'

        # Filter the list to include only files with     the desired extension
        txt_files = [file for file in files if os.path.splitext(file)[1] == image_ext]
        json_files = [json_files for json_files in files if os.path.splitext(json_files)[1] == json_ext]

        for text_file in txt_files:
            txt_split = text_file.split('.')[0]
            for json_file in json_files:
                json_split = json_file.split('.')[0]
                if txt_split == json_split:
                    shutil.copy(f'{data_dir}/{text_file}', f'{destination_path}/images')
                    shutil.copy(f'{data_dir}/{json_file}', f'{destination_path}/annotation_json')
    except Exception as e:
        _logger.error(f'image and json file matching validation process failed, due to {e}')


def write_to_xml(image_name, bboxes, image_folder_name, coco_data, save_folder, database_name=''):
    try:
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
    except Exception as e:
        _logger.error(f'XML file creation process failed, due to {str(e)}')
        raise Exception(str(e))


def xml_dataset_preparation(args):
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
        raise Exception(str(e))


def width_and_height_management(json_dir):
    unwanted_points = []
    try:
        if os.path.exists(json_dir) and os.listdir(json_dir):
            for annotate_file in os.listdir(json_dir):
                with open(os.path.join(json_dir, annotate_file), 'r') as file:
                    coco_file = json.load(file)
                for shapes in coco_file['shapes']:
                    if shapes['points'][0][0] > shapes['points'][1][0]:
                        shapes['points'][0][0], shapes['points'][1][0] = shapes['points'][1][0], shapes['points'][0][0]
                    if shapes['points'][0][1] > shapes['points'][1][1]:
                        shapes['points'][0][1], shapes['points'][1][1] = shapes['points'][1][1], shapes['points'][0][1]
                    if shapes['points'][0][0] == shapes['points'][1][0] or shapes['points'][0][1] == \
                            shapes['points'][1][1]:
                        unwanted_points.append(shapes)
                if unwanted_points:
                    [coco_file['shapes'].remove(waste_points) for waste_points in unwanted_points]
                with open(r"C:\Users\VR DELLA\Downloads\00610_lr.json", 'w') as f:
                    json.dump(coco_file, f)
    except Exception as e:
        _logger.error(f'Width & Height validate process failed, due to {str(e)}')
        raise Exception(str(e))


parser = argparse.ArgumentParser(description='This method is for convert the json annotation file to xml file for rcnn')

parser.add_argument('--entire_path', required=False, help='Folder path with all the annotated images and corresponding json')
parser.add_argument('--json_data', required=False, help='This folder of json data which is going to convert')
parser.add_argument('--save_location', required=False, help='The folder name and path which is converted file going to store')
parser.set_defaults(skip_background=True)

args = parser.parse_args()


def process_main_script(args):
    try:
        if args.entire_path:
            destination_path = os.path.join(os.path.dirname(args.entire_path), 'converted_dataset')
            if not os.path.exists(destination_path):
                os.mkdir(destination_path)
                os.mkdir(os.path.join(destination_path, 'annotation_json'))
                os.mkdir(os.path.join(destination_path, 'images'))
                args.json_data = os.path.join(destination_path, 'annotation_json')
            image_and_json_file_validation(args.entire_path, destination_path)
        if not args.save_location:
            os.mkdir(os.path.join(os.path.dirname(args.json_data), 'converted_xml'))
            args.save_location = os.path.join(os.path.dirname(args.json_data), 'converted_xml')
        width_and_height_management(args.json_data)
        xml_dataset_preparation(args)
        _logger.info('xml file creation process completed successfully')
    except Exception as e:
        _logger.error(f'xml file creation process has failed!!...., due to {e}')


process_main_script(args)
