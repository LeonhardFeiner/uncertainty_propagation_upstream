import torch
import fastmri.losses

class BaseNetwork(torch.nn.Module):
    def optimizer(self):
        return torch.optim.Adam

    def scheduler(self):
        return torch.optim.lr_scheduler.ReduceLROnPlateau

    def criterion(self):
        return fastmri.losses.MaskedL1Loss

    @property
    def log_dict(self):
        return {}

    def preprocess(self, inp):
        return inp, {}

    def postprocess(self, inp, **kwargs):
        return inp