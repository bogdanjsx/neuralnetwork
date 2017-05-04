import numpy as np

def im2col(input, size, stride = 1):
    # Parameters
    D, W, H = input.shape
    col_ext = H - size + 1
    row_ext = W - size + 1

    # Starting index
    start_idx = np.arange(size)[:, None] * H + np.arange(size)

    # Depth index
    depth_idx = W * H * np.arange(D)

    # Starting index with depth
    start_idx = (depth_idx[:, None] + start_idx.flatten()).reshape((-1, size, size))

    # Offset index
    offset_idx = np.arange(col_ext) + np.arange(row_ext)[:, None] * H

    # Final index (to be used with np.take from input array)
    res = offset_idx.flatten()[:, None] + start_idx.flatten()

    # Filter the list if greater stride
    if stride > 1:
        idx = [i for i in range(res.shape[0]) if (i % col_ext) % stride == 0 and (i // col_ext) % stride == 0]
        return np.take(res, idx, axis=0)

    return res

def col2im(cols, values, xshape):
    # Returns an array with the given values at the 'cols' indices
    res = np.zeros(xshape).flatten()

    np.add.at(res, cols.flatten(), values.flatten())
    return res.reshape(xshape)
