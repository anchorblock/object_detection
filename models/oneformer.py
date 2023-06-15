from transformers.models.oneformer.modeling_oneformer import *


class ModifiedOneFormerForUniversalSegmentation(OneFormerForUniversalSegmentation):
    main_input_name = ["pixel_values", "task_inputs"]

    def __init__(self, config: OneFormerConfig, backbone = None):
        super().__init__(config)
        self.model = OneFormerModel(config)

        self.matcher = OneFormerHungarianMatcher(
            cost_class=config.class_weight,
            cost_dice=config.dice_weight,
            cost_mask=config.mask_weight,
            num_points=config.train_num_points,
        )

        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
            "loss_contrastive": config.contrastive_weight,
        }

        self.criterion = OneFormerLoss(
            num_classes=config.num_labels,
            matcher=self.matcher,
            weight_dict=self.weight_dict,
            eos_coef=config.no_object_weight,
            num_points=config.train_num_points,
            oversample_ratio=config.oversample_ratio,
            importance_sample_ratio=config.importance_sample_ratio,
            contrastive_temperature=config.contrastive_temperature,
        )

        self.post_init()

        # loading pretrained backbone weights
        if backbone is not None:
            self.model.pixel_level_module.encoder.load_state_dict(backbone.state_dict(), strict=False)