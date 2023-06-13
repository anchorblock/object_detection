import json

# Load coco_labels_raw.json
with open('coco_labels_raw.json', 'r') as f:
    coco_labels_raw = json.load(f)

##### COCO PANOPTIC

# Create coco-panoptic-id2label.json dictionary
coco_panoptic_id2label = {}
for label_id, label_info in coco_labels_raw.items():
    new_id = label_info['new_id']
    name = label_info['name']
    coco_panoptic_id2label[new_id] = name

# Save coco-panoptic-id2label.json
with open('coco-panoptic-id2label.json', 'w') as f:
    json.dump(coco_panoptic_id2label, f, indent=4)

# Create coco-panoptic-label2id dictionary
coco_panoptic_label2id = {v: k for k, v in coco_panoptic_id2label.items()}

# Save coco-panoptic-label2id.json
with open('coco-panoptic-label2id.json', 'w') as f:
    json.dump(coco_panoptic_label2id, f, indent=4)



##### COCO DETECTION

# Create coco-detection-id2label.json dictionary
coco_detection_id2label = {}
id_number = 0
for label_id, label_info in coco_labels_raw.items():
    if label_info['isthing'] == 0:
        continue
    name = label_info['name']
    coco_detection_id2label[id_number] = name
    id_number += 1

# Save coco-detection-id2label.json
with open('coco-detection-id2label.json', 'w') as f:
    json.dump(coco_detection_id2label, f, indent=4)

# Create coco-detection-label2id dictionary
coco_detection_label2id = {v: k for k, v in coco_detection_id2label.items()}

# Save coco-detection-label2id.json
with open('coco-detection-label2id.json', 'w') as f:
    json.dump(coco_detection_label2id, f, indent=4)