from maskay.tensorsat import TensorSat


class Predictor:
    def __init__(
        self,
        cropsize: int = 512,
        overlap: int = 32,
        device: str = "cpu",
        batchsize: int = 1,
        quiet: int = False,
    ):
        self.cropsize = cropsize
        self.overlap = overlap
        self.device = device
        self.batchsize = batchsize
        self.quiet = quiet
        self.result = None

    def predict(self, model, tensor: TensorSat):
        model.cropsize = self.cropsize
        model.overlap = self.overlap
        model.device = self.device
        model.batchsize = self.batchsize
        model.quiet = self.quiet
        self.result = model._predict(tensor=tensor)
        return self.result

    def __repr__(self) -> str:
        # no long line
        msg = (
            "Predictor("
            + "classes="
            + str(self.classes)
            + ", "
            + "cropsize="
            + str(self.cropsize)
            + ", "
            + "overlap="
            + str(self.overlap)
            + ", "
            + "device="
            + str(self.device)
            + ", "
            + "batchsize="
            + str(self.batchsize)
            + ", "
            + "quiet="
            + str(self.quiet)
            + ")"
        )

        return msg

    def __str__(self) -> str:
        return self.__repr__()
