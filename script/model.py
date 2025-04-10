import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def create_unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
):
    """
    Create a UNet model using segmentation_models_pytorch

    Args:
        encoder_name (str): Name of the encoder model to use (default: 'resnet34')
        encoder_weights (str): Pre-trained encoder weights (default: 'imagenet')
        in_channels (int): Number of input channels (default: 3)
        classes (int): Number of output classes (default: 1)
        activation (str): Activation for the final layer (default: None)

    Returns:
        model: UNet model
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation
    )

    return model


class AVSegmentationModel(nn.Module):
    """
    Wrapper model for AV segmentation
    """

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,  # Single class for either artery or vein segmentation
    ):
        super(AVSegmentationModel, self).__init__()

        self.model = create_unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        """
        Predict function with sigmoid activation for binary segmentation
        """
        with torch.no_grad():
            x = self.forward(x)
            return torch.sigmoid(x)
