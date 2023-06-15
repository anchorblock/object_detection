from transformers.models.mask2former.modeling_mask2former import *
from transformers import MaskFormerSwinConfig


class CustomMask2FormerConfig(Mask2FormerConfig):
    model_type = "custom_mask2former"
    attribute_map = {"hidden_size": "hidden_dim"}
    backbones_supported = ['bit', 'convnext', 'convnextv2', 'dinat', 'focalnet', 'nat', 'resnet', 'swin']



class CustomMask2FormerPixelLevelModule(Mask2FormerPixelLevelModule):
    def __init__(self, config: CustomMask2FormerConfig):
        """
        Pixel Level Module proposed in [Masked-attention Mask Transformer for Universal Image
        Segmentation](https://arxiv.org/abs/2112.01527). It runs the input image through a backbone and a pixel
        decoder, generating multi-scale feature maps and pixel embeddings.

        Args:
            config ([`CustomMask2FormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__(config)

        backbone_config = config.backbone_config

        if backbone_config.model_type == "swin":
            # for backwards compatibility
            backbone_config = MaskFormerSwinConfig.from_dict(backbone_config.to_dict())
            
            backbone_config.out_features = ["stage1", "stage2", "stage3", "stage4"]

        backbone_config.out_features = ["stage1", "stage2", "stage3", "stage4"]
        self.encoder = AutoBackbone.from_config(backbone_config)
        self.decoder = Mask2FormerPixelDecoder(config, feature_channels=self.encoder.channels)



class CustomMask2FormerModel(Mask2FormerModel):
    main_input_name = "pixel_values"

    def __init__(self, config: CustomMask2FormerConfig):
        super().__init__(config)
        self.pixel_level_module = CustomMask2FormerPixelLevelModule(config)
        self.transformer_module = Mask2FormerTransformerModule(in_features=config.feature_size, config=config)

        self.post_init()



class CustomMask2FormerForUniversalSegmentation(Mask2FormerForUniversalSegmentation):
    main_input_name = "pixel_values"

    def __init__(self, config: CustomMask2FormerConfig, backbone = None):
        super().__init__(config)
        self.model = CustomMask2FormerModel(config)

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

