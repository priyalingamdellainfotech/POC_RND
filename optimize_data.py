import json
import os

data_dir = r'C:\Users\VR DELLA\Downloads\Lead2_json_files 1\Lead2_json_files'

label_list = ['P_Wave', 'T_Wave', 'QRS_Complex']

inside_label = []

for i_data in os.listdir(data_dir):
    print(i_data)
    if i_data.endswith('.json'):
        with open(os.path.join(data_dir, i_data), 'r') as f:
            coco_data = json.load(f)
        for labels in coco_data['shapes']:
            inside_label.append(labels['label'])
        # for labels in coco_data['shapes']:
        #     if labels['label'] == 'p wave':
        #         labels['label'] = labels['label'].replace('p wave', label_list[0])
        #     if labels['label'] == 'T wave':
        #         labels['label'] = labels['label'].replace('T wave', label_list[1])
        #     if labels['label'] == 'qrs':
        #         labels['label'] = labels['label'].replace('qrs', label_list[2])
        # with open(os.path.join(data_dir, i_data), 'w') as f:
        #     json.dump(coco_data, f)
unique_labels = list(set(inside_label))
print(unique_labels)
