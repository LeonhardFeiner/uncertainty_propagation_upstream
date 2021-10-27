import torch

def center_crop(data, shape, channel_last=False):
    """
    [source] https://github.com/facebookresearch/fastMRI/blob/master/data/transforms.py
    Apply a center crop to the input real image or batch of real images.

    Args:
        data (numpy.array): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        numpy.array: The center cropped image
    """
    if channel_last:
        dim0 = data.shape[-3]
        dim1 = data.shape[-2]
    else:
        dim0 = data.shape[-2]
        dim1 = data.shape[-1]

    assert 0 < shape[0] <= dim0
    assert 0 < shape[1] <= dim1
    w_from = (dim0 - shape[0]) // 2
    h_from = (dim1 - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    if channel_last:
        return data[..., w_from:w_to, h_from:h_to, :]
    else:
        return data[..., w_from:w_to, h_from:h_to]

def complex2real(z, channel_last=False):
    stack_dim = -1 if channel_last else 1
    return torch.cat([torch.real(z), torch.imag(z)], stack_dim)

def real2complex(z, channel_last=False):
    stack_dim = -1 if channel_last else 1
    (real, imag) = torch.chunk(z, 2, axis=stack_dim)
    return torch.complex(real, imag)
