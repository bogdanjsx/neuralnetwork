import numpy as np

def im2col(input, size, stepsize=1):
    # Parameters
    D, W, H = input.shape
    col_ext = H - size[1] + 1
    row_ext = W - size[0] + 1

    # Starting index
    start_idx = np.arange(size[0])[:, None] * H + np.arange(size[1])

    # Depth index
    depth_idx = W * H * np.arange(D)

    # Starting index with depth
    start_idx = (depth_idx[:, None] + start_idx.ravel()).reshape((-1, size[0], size[1]))

    # Offset index
    offset_idx = np.arange(col_ext) + np.arange(row_ext)[:, None] * H

    # Final index (to be used with np.take from input array)
    return offset_idx.ravel()[:, None] + start_idx.ravel()


def col2im(cols, values, xshape):
  res = np.zeros(xshape).flatten()
  np.add.at(res, cols.flatten(), values.flatten())
  return res.reshape(xshape)


def col2im_good(im2col_mat, bsz, img_sz):
    row_nr, col_nr = im2col_mat.shape
    row_nr /= 3

    b_rows, b_cols = bsz
    img_rows, img_cols = img_sz
    img = np.zeros((img_rows, img_cols, 3))

    #
    start_idx = []
    for row in range(0, img_rows - b_rows + 1, 1):
        for col in range(0, img_cols - b_cols + 1, 1):
            start_idx.append((row, col))

    #print img

    im2col_mat_col = 0
    for s_idx in start_idx:
        row, col = s_idx
        #print "#" + str(row) + " " + str(col) + "#"

        im2col_mat_row = 0
        for br in range(b_rows):
            for bc in range(b_cols):
                img[row+br][col+bc][0] = \
                    im2col_mat[im2col_mat_row][im2col_mat_col]
                img[row+br][col+bc][1] = im2col_mat[im2col_mat_row+1][im2col_mat_col]
                img[row+br][col+bc][2] = im2col_mat[im2col_mat_row+2][im2col_mat_col]
                im2col_mat_row += 3
        im2col_mat_col += 1
    return img