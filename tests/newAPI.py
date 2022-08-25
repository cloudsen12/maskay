def predict(
    tensor: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
    cropsize: int = 512,
    overlap: int = 128,
    batch_size: int = 1,
    model: Union[str, Model] = "UnetMobV2",
    device: str = "cpu",
    quiet: bool = False,
):
    # Number of channels (C)
    CHANNELS = tensor.shape[0]

    ## Getting global parameters -----------------------------
    # if the model is str get it from the library subpackage
    if isinstance(model, str):
        TModel = eval("%s(quiet=%s)" % (model, quiet))

    if not isinstance(TModel, Model):
        raise ValueError("The model argument must be a maskay.Model dataclass")

    # Set the model in cpu or gpu device
    if TModel.framework == "torch":
        from .run.TorchRun import TorchRun

        if device == "cuda":
            TModel.callable = TModel.callable.cuda()

        # Run the model under evaluation mode
        TModel.callable.eval()
    elif TModel.framework == "tensorflow":
        from .run.TensorFlowRun import TensorFlowRun
    else:
        raise ValueError("DL Framework not supported")

    # Select only the bands to use
    if TModel.bands != "ALL":
        bands = np.array(TModel.bands)
        tensor = tensor[:, bands, :, :]

    # Identify the output data type default (np.float32)
    dtype = TModel.outfunction(np.random.rand(1)).dtype

    # Create the output tensor
    outensor = np.zeros(
        shape=(TModel.output_bands, tensor.shape[1], tensor.shape[2])
    ).astype(dtype)

    # Change the order of the tensor to (B, Hip, Wip, C) if necessary
    if TModel.order == "BHWC":
        tensor = np.moveaxis(tensor, 0, -1)

    # Chop the tensor into smaller image patches
    mrs = __MagickCrop(tensor=tensor, cropsize=cropsize, overlap=overlap, quiet=quiet)

    # Run by image patches
    iter = np.arange(0, len(mrs), batch_size)
    for index in tqdm(iter, disable=quiet):
        # Create a Image Patch tensor BCHW
        IPtemp = np.zeros(shape=(batch_size, CHANNELS, cropsize, cropsize)).astype(
            tensor.dtype
        )
        batch_step = mrs[index : (index + batch_size)]
        for index2, mr in enumerate(batch_step):
            IPtemp[index2] = tensor[
                None, :, mr[0] : (mr[0] + cropsize), mr[1] : (mr[1] + cropsize)
            ]
        IP = IPtemp

        # Run the InFunction (preprocessing)
        IP = TModel.infunction(IP)

        # Run the model
        IP = TorchRun(TModel.callable, IP, device)

        # Run the OutFunction
        IP = TModel.outfunction(IP)

        # Save the output in the outensor
        mrsTemps = __MagickGather(outensor, batch_step, overlap, cropsize)

        for index in range(len(IP)):
            (Xmin, Ymin), (Xmax, Ymax) = mrsTemps[index]["outensor"]
            (XIPmin, YIPmin), (XIPmax, YIPmax) = mrsTemps[index]["ip"]
            # Put the IP tensor in the output tensor
            outensor[:, Xmin:Xmax, Ymin:Ymax] = IP[
                index, :, XIPmin:XIPmax, YIPmin:YIPmax
            ]

    return outensor
