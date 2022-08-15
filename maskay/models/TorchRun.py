import torch
import numpy as np
from tqdm import tqdm
from ..utils import color


def TorchRun(model, IPs, tensorprob, bands, batch_size, order, device, quiet):
    # Convert numpy array to tensor
    if bands != "ALL":
        bands = np.array(bands) - 1
        IPs = IPs[:, bands, :, :]        
    IPs = torch.Tensor(IPs)
    
    # Change the order of the tensor to (B, Hip, Wip, C) if necessary
    if order == "BHWC":
        IPs = torch.transpose(IPs, perm=[0, 2, 3, 1])
    
    # Run the model on every image patch
    if not quiet:
        print(color.BLUE + color.BOLD + 
            " Running the SegModel on every image patch ..." + color.END
        )
        
    bsRange = np.arange(0, IPs.shape[0], batch_size)
    for index in tqdm(bsRange, disable=quiet):
        with torch.no_grad():
            IPbatch = IPs[index:(index + batch_size)]
            if device == "cuda":
                IPbatch = IPbatch.cuda()
            tensorprob[index:(index+batch_size)] = model(IPbatch).detach().cpu().numpy()
        torch.cuda.empty_cache()
        
    if not quiet:
        print("")
    
    return tensorprob