import json

# Load imagenet-1k_labels_raw.json
with open('imagenet-1k_labels_raw.json', 'r') as f:
    imagenet_labels_raw = json.load(f)

##### IMAGNENET-1k (2012)

# Create imagenet-1k-id2label.json dictionary
imagenet_id2label = {}
for new_id, (label_id, label_info) in enumerate(imagenet_labels_raw.items()):
    imagenet_id2label[new_id] = f"{label_id} - {label_info}"

# Save imagenet-1k-id2label.json
with open('imagenet-1k-id2label.json', 'w') as f:
    json.dump(imagenet_id2label, f, indent=4)

# Create imagenet-label2id dictionary
imagenet_label2id = {v: k for k, v in imagenet_id2label.items()}

# Save imagenet-1k-label2id.json
with open('imagenet-1k-label2id.json', 'w') as f:
    json.dump(imagenet_label2id, f, indent=4)


