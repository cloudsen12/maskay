import torch
import maskay.models


def predict_tensor(tensor: torch.Tensor, device: str = "cuda", model: str = "adan") -> torch.Tensor:
    """Predict cloud cover from a SEN2 tensor.

    Args:
        tensor (torch.Tensor): A SEN2 with shape (B, C, H, W).
        device (str, optional): The device to use. Defaults to "cuda".
        model (str, optional): Define the key of the model to use. Defaults to "adan".

    Raises:
        ValueError: If the model is not supported.

    Returns:

        torch.Tensor: A tensor with shape (B, 4, H, W) with the cloud cover probabilities.
    """
    pass

def predict_SAFE(folder:str, output: str, device: str = "cuda", model: str = "adan") -> bool:
    """Predict cloud cover from a SAFE folder.
        
    Args:
        folder (str): The path to the SAFE folder.
        device (str, optional): The device to use. Defaults to "cuda".
        model (str, optional): Define the key of the model to use. Defaults to "adan".
        output (str): The output directory.
        
    Raises:
        ValueError: If the model is not supported.

    Returns:
        bool: True if a GeoTIFF was created.
    """    
    pass


def __predict_tensor512(tensor: torch.Tensor, device: str = "cuda", model: str = "adan") -> torch.Tensor:
    """Predict cloud cover from a SEN2 tensor.

    Args:
        tensor (torch.Tensor): A SEN2 with shape (1, CHANNELS, 512, 512).
        device (str, optional): The device to use. Defaults to "cuda".
        model (str, optional): Define the key of the model to use. Defaults to "adan".

    Raises:
        ValueError: If the model is not supported.

    Returns:
        torch.Tensor: A tensor with shape (4, 512, 512) with the cloud cover probabilities.
    """
    if model == "adan":
        prediction = maskay.models.adan.AdanPredict(tensor, device)
    else:
        # raise an error
        raise ValueError("Model not supported")
    return prediction


if __name__ == "__main__":
    tensor = torch.randn(1, 13, 512, 512)
    cloudprob = predict(tensor)
