def squeeze_batchdim(inputs):
    if isinstance(inputs, list):
        return [inp.squeeze(0) for inp in inputs]
    else:
        return inputs.squeeze(0)

def to_device(inputs, device):
    if isinstance(inputs, list):
        return [inp.to(device) for inp in inputs]
    else:
        return inputs.to(device)

def prepare_batch(batch, device):
    inputs, outputs = batch
    return to_device(squeeze_batchdim(inputs), device), to_device(squeeze_batchdim(outputs), device)