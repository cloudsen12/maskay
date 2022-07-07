import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch
import pathlib
import requests

class CloudModelAdan(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=13,
            classes=4, 
        )
    def forward(self, x):
        return self.model(x)
    

def AdanPredict(tensor: torch.Tensor, device: str = "cuda"):
    
    __model__ = "https://zenodo.org/record/6807536/files/lilith.ckpt"
    
    # if file does not exist, download it!
    if not pathlib.Path("weights/adam.ckpt").is_file():        
        r = requests.get(__model__, allow_redirects=True)
        with open('weights/adam.ckpt', 'wb') as f:
            f.write(r.content)
    
    # mage a prediction using a pretrained model
    model = CloudModelAdan()
    model = model.load_from_checkpoint("weights/adam.ckpt")
    model.eval()
    
    if device == "cuda":
        model = model.cuda()
        tensor = tensor.cuda()
        preds = model.forward(tensor)        
    elif device == "cpu":
        preds = model.forward(tensor)
        
    # prediction = torch.argmax(preds, axis=1).detach().cpu().numpy()
    
    return preds.squeeze().detach().cpu().numpy()
