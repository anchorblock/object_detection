from transformers import OneFormerConfig, OneFormerModel

# Initializing a OneFormer shi-labs/oneformer_ade20k_swin_tiny configuration
configuration = OneFormerConfig()

print(configuration)

print("\n\n\n")
# Initializing a model (with random weights) from the shi-labs/oneformer_ade20k_swin_tiny style configuration
model = OneFormerModel(configuration)

print(model)
print("\n\n\n")


# Accessing the model configuration
configuration = model.config

print(configuration)


##########################################################


class Mask2FormerSwinLarge(nn.Module):
    def __init__(self, num_classes):
        super(Mask2FormerSwinLarge, self).__init__()
        self.backbone = swin_transformer.sWinTransformer(
            img_size=384,
            patch_size=4,
            in_chans=3,
            num_classes=0  # Set to 0 to exclude the original classification head
        )
        self.classifier = nn.Linear(1024, num_classes)  # Modify the number of output classes as per your requirement

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)

        return x
