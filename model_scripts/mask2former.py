
        backbone_config_dict = config.backbone_config.to_dict()
        backbone_config = SwinConfig.from_dict(backbone_config_dict)

        self.encoder = AutoBackbone.from_config(backbone_config)
        self.decoder = Mask2FormerPixelDecoder(config, feature_channels=self.encoder.channels)