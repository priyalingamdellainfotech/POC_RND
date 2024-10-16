import os

json_dir = r"C:\Users\VR DELLA\Downloads\vehicle images 3\vehicle images"
image_dir = r"C:\Users\VR DELLA\Downloads\vehicle images 3\rearrange_files"
files = os.listdir(json_dir)
image_dirs = os.listdir(image_dir)
try:
    for idx, file in enumerate(image_dirs):
        for i_file in files:
            if file.split('.')[0] == i_file.split('.')[0]:
                os.rename(os.path.join(json_dir, i_file), os.path.join(json_dir, f"{idx}.{i_file.split('.')[1]}"))
                os.rename(os.path.join(image_dir, file), os.path.join(image_dir, f"{idx}.{file.split('.')[1]}"))
except Exception as e:
    print(e)