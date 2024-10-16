import os
import shutil

data_dir = r'C:\Users\VR DELLA\Downloads\unique_images\unique_images'
destination_path = r'C:\Users\VR DELLA\Desktop\train2017\train2017'

files = os.listdir(data_dir)
image_ext = '.png'
json_ext = '.txt'

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

print('success')
