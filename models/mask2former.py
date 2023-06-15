from transformers.models.mask2former.modeling_mask2former import *


class ModifiedMask2FormerForUniversalSegmentation(Mask2FormerForUniversalSegmentation):
    main_input_name = "pixel_values"

    def __init__(self, config: Mask2FormerConfig, backbone = None):
        super().__init__(config)
        self.model = Mask2FormerModel(config)

        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.class_predictor = nn.Linear(config.hidden_dim, config.num_labels + 1)

        self.criterion = Mask2FormerLoss(config=config, weight_dict=self.weight_dict)

        self.post_init()

        # loading pretrained backbone weights
        if backbone is not None:
            self.model.pixel_level_module.encoder.load_state_dict(backbone.state_dict(), strict=False)
