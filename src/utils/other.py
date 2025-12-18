import torch
import math
import warnings
import sys

def identity(n, sparse=False, device=None):
    if sparse:
        indices = torch.arange(n).unsqueeze(0).repeat(2, 1)
        values = torch.ones(n)
        return torch.sparse_coo_tensor(indices, values, (n, n), device=device)
    else:
        return torch.eye(n, device=device)


def conv2d_calc_out_size(X, kernel_size, stride=1, padding=0, dilation=1, transposed=False):
    N, C, H, W = X.shape

    if isinstance(padding, tuple):
        pad_left, pad_right, pad_top, pad_bottom = padding
    else:
        pad_left = pad_right = pad_top = pad_bottom = padding

    if not transposed:  # conv2d
        width  = (W + (pad_left + pad_right) - dilation * (kernel_size - 1) - 1) / stride + 1
        height = (H + (pad_top + pad_bottom) - dilation * (kernel_size - 1) - 1) / stride + 1
    else:               # conv_transpose2d
        width  = (W - 1) * stride - (pad_left + pad_right) + dilation * (kernel_size - 1) + 1
        height = (H - 1) * stride - (pad_top + pad_bottom) + dilation * (kernel_size - 1) + 1

    if width != int(width) or height != int(height):
        if 'pytest' not in sys.modules:  # hide the warning at test time
            warnings.warn(f'Caution: Input{list(X.shape)} - The expected output size after convolution ({width:.1f}x{height:.1f}) is not an integer. Consider adjusting stride/padding/kernel to get an integer output size. Using rounded value: {int(width)}x{int(height)}')
    return int(height), int(width)

def conv2d_pad_string_to_int(padding, kernel_size):
    if padding == 'valid':
        return 0
    elif padding == 'same':
        assert kernel_size % 2 == 1, f'For "same" padding the kernel_size is expected to be odd number, but got: {kernel_size}'
        return (kernel_size - 1) // 2
    elif padding == 'full':
        return kernel_size - 1
    else:
        return padding


def paddings_mask(lengths, max_len):
    target_shape = (*lengths.shape, max_len)
    mask = torch.arange(max_len).expand(target_shape) >= lengths.unsqueeze(-1)
    return mask


def sparse_slice(x, dim, end):
    x = x.coalesce()  # it's very important to sum the values of possibly duplicated indices
    indices, values = x.indices(), x.values()
    mask = indices[dim] < end
    return torch.sparse_coo_tensor(indices[:, mask], values[mask])


def to_power_of_2(value, max_multiple=256):
    multiple = min(max_multiple, 2 ** int(math.log2(value)))
    return round(value / multiple) * multiple


def format_seconds(seconds):
    seconds = round(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours}h {minutes}m {seconds}s"


# Preserves public methods when compiling non-nn.Module objects
def custom_compile(model, wrap_methods):
    compiled = torch.compile(model)
    for method in wrap_methods:
        setattr(compiled, method, getattr(model, method))
    return compiled


def chunk_equal(arr, groups, pad_symbol=None):
    n, left = divmod(len(arr), groups)
    max_size = n + (1 if left else 0)

    chunks = []
    step = 0
    for i in range(groups):
        size = n + int(i < left)
        part = arr[step : step + size]
        pad = [pad_symbol] * (max_size - size)
        chunks.append(part + pad)
        step += size

    return chunks

