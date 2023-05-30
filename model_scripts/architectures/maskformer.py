        backbone_config = config.backbone_config
        if backbone_config.model_type == "swin":
            # for backwards compatibility
            backbone_config = MaskFormerSwinConfig.from_dict(backbone_config.to_dict())
            backbone_config.out_features = ["stage1", "stage2", "stage3", "stage4"]
        self.encoder = AutoBackbone.from_config(backbone_config)