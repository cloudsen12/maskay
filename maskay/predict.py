import torch
import maskay.models

def predict(tensor: torch.Tensor, device: str="cuda", model: str="adan"):
    """ Predict cloud cover from a SEN2 tensor.

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