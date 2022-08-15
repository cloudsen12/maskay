import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import pathlib
from ...utils import get_models_path


class UnetMobV2Class(pl.LightningModule):
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
    #    
    #def set_maskay_params(self, maskayparams):
    #    self.maskayparams = maskayparams
        
def UnetMobV2():
    filename = (pathlib.Path(get_models_path()) / "unetmobv2.ckpt").as_posix()
    model = UnetMobV2Class().load_from_checkpoint(filename)
    model = model.eval()
    return model