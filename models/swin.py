

needs:

 self.model = MaskFormerModel(config)

 hidden_size = config.decoder_config.hidden_size

self.class_predictor = nn.Linear(hidden_size, config.num_labels + 1)

 self.mask_embedder = MaskformerMLPPredictionHead(hidden_size, hidden_size, config.mask_feature_size)

self.matcher = MaskFormerHungarianMatcher(
            cost_class=1.0, cost_dice=config.dice_weight, cost_mask=config.mask_weight
        )

def get_logits(



###############################
 #    focalnet + mask2former


import torch
from transformers import FocalNetForImageClassification, FocalNetConfig
from mask2former.modeling import Mask2Former


model = Mask2Former.from_pretrained('MODEL_NAME')
config = model.config


# Load the pre-trained FocalNet model
focalnet_model = FocalNetForImageClassification.from_pretrained('focalnet_model_name')
focalnet_config = focalnet_model.config

# Replace the backbone with the FocalNet model
model.backbone = focalnet_model.backbone
model.backbone.config = focalnet_config
model.backbone.model_name = 'focalnet_model_name'


# freeze

for param in model.backbone.parameters():
    param.requires_grad = False



# other training hyperparameters

# Modify other training hyperparameters as needed
model.config.learning_rate = 0.001
model.config.num_train_epochs = 5







