import tensorflow as tf
import numpy as np
from tqdm import tqdm
from ..utils import color


def TensorFlowRun(model, IPs, tensorprob, bands, batch_size, order, device, quiet):
    # Convert numpy array to tensor
    if bands != "ALL":
        bands = np.array(bands) - 1
        IPs = IPs[:, bands, :, :]        
    IPs = tf.cast(IPs, dtype=tf.float32)
    
    # Change the order of the tensor to (B, Hip, Wip, C) if necessary
    if order == "BHWC":
        IPs = tf.transpose(IPs, perm=[0, 2, 3, 1])
        
    # Run the model on every image patch
    if not quiet:
        print(color.BLUE + color.BOLD + 
            " Running the DLModel on every image patch ..." + color.END
        )
    
    # Run the model for batch size
    bsRange = np.arange(0, IPs.shape[0], batch_size)
    
    for index in tqdm(bsRange, disable=quiet):
        IPbatch = IPs[index:(index + batch_size)]
        if device == "cuda":
            with tf.device('/GPU:0'):
                IPprediction = tf.nn.softmax(model(IPbatch)).numpy()
                IPprediction = np.moveaxis(IPprediction, 3, 1)
                tensorprob[index:(index+batch_size)] = IPprediction
        elif device == "cpu":
            with tf.device('/CPU:0'):
                IPprediction = tf.nn.softmax(model(IPbatch)).numpy()
                IPprediction = np.moveaxis(IPprediction, 3, 1)
                tensorprob[index:(index+batch_size)] = IPprediction
            
    return tensorprob