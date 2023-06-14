from transformers.models.maskformer.modeling_maskformer import *


class CustomMaskFormerConfig(MaskFormerConfig):
    model_type = "custom_maskformer"
    attribute_map = {"hidden_size": "mask_feature_size"}
    backbones_supported = ['bit', 'convnext', 'convnextv2', 'dinat', 'focalnet', 'nat', 'resnet', 'swin']
    decoders_supported = ["detr"]



class CustomMaskFormerPixelLevelModule(MaskFormerPixelLevelModule):

    def __init__(self, config: CustomMaskFormerConfig):
        """
        Pixel Level Module proposed in [Per-Pixel Classification is Not All You Need for Semantic
        Segmentation](https://arxiv.org/abs/2107.06278). It runs the input image through a backbone and a pixel
        decoder, generating an image feature map and pixel embeddings.

        Args:
            config ([`CustomMaskFormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__(config)

        # TODD: add method to load pretrained weights of backbone
        backbone_config = config.backbone_config
        if backbone_config.model_type == "swin":
            # for backwards compatibility
            backbone_config = MaskFormerSwinConfig.from_dict(backbone_config.to_dict())
            backbone_config.out_features = ["stage1", "stage2", "stage3", "stage4"]

        backbone_config.out_features = ["stage1", "stage2", "stage3", "stage4"]
        self.encoder = AutoBackbone.from_config(backbone_config)

        feature_channels = self.encoder.channels
        self.decoder = MaskFormerPixelDecoder(
            in_features=feature_channels[-1],
            feature_size=config.fpn_feature_size,
            mask_feature_size=config.mask_feature_size,
            lateral_widths=feature_channels[:-1],
        )

    def forward(self, pixel_values: Tensor, output_hidden_states: bool = False) -> MaskFormerPixelLevelModuleOutput:
        features = self.encoder(pixel_values).feature_maps
        decoder_output = self.decoder(features, output_hidden_states)
        return MaskFormerPixelLevelModuleOutput(
            # the last feature is actually the output from the last layer
            encoder_last_hidden_state=features[-1],
            decoder_last_hidden_state=decoder_output.last_hidden_state,
            encoder_hidden_states=tuple(features) if output_hidden_states else (),
            decoder_hidden_states=decoder_output.hidden_states if output_hidden_states else (),
        )


class CustomMaskFormerModel(MaskFormerModel):

    def __init__(self, config: CustomMaskFormerConfig):
        super().__init__(config)
        self.pixel_level_module = CustomMaskFormerPixelLevelModule(config)
        self.transformer_module = MaskFormerTransformerModule(
            in_features=self.pixel_level_module.encoder.channels[-1], config=config
        )

        self.post_init()


class CustomMaskFormerForInstanceSegmentation(MaskFormerForInstanceSegmentation):

    def __init__(self, config: CustomMaskFormerConfig):
        super().__init__(config)
        self.model = CustomMaskFormerModel(config)
        hidden_size = config.decoder_config.hidden_size
        # + 1 because we add the "null" class
        self.class_predictor = nn.Linear(hidden_size, config.num_labels + 1)
        self.mask_embedder = MaskformerMLPPredictionHead(hidden_size, hidden_size, config.mask_feature_size)

        self.matcher = MaskFormerHungarianMatcher(
            cost_class=1.0, cost_dice=config.dice_weight, cost_mask=config.mask_weight
        )

        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.cross_entropy_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.criterion = MaskFormerLoss(
            config.num_labels,
            matcher=self.matcher,
            weight_dict=self.weight_dict,
            eos_coef=config.no_object_weight,
        )

        self.post_init()
