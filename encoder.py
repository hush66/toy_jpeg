from PIL import Image
from utils import *
from huffman import HuffmanTree
import numpy as np


def main():
    image = Image.open(IMAGE_PATH)
    ycbcr = image.convert('YCbCr')

    matrix = np.array(ycbcr)
    row, col = matrix.shape[0], matrix.shape[1]

    # 8*8 partition
    if row%8 == col%8 == 0:
        blocks = (row // 8) * (col // 8)
    else:
        raise ValueError(("The image size wrong!"))

    # store ac and dc ingredient
    dc = np.empty((blocks, 3), dtype=np.int32)
    ac = np.empty((blocks, 63, 3), dtype=np.int32)

    block_index = 0
    for i in range(0, row, 8):
        for j in range(0, col, 8):
            for k in range(3):
                # split 8*8 block
                # change data range from [0,255] to [-128,127] due to DCT changes, domain symmetry is required
                cur_block = matrix[i:i+8, j:j+8, k] - 128

                # FDCT
                dct_block = DCT2D(cur_block)

                # quantization type:1 use Q_y type:2 use Q_c
                quantized_block = quantize(dct_block, 1)

                # zigzag scan
                zigzag = zigzag_scan(quantized_block)

                # get dc and ac ingredient
                dc[block_index, k] = zigzag[0]
                ac[block_index, :, k] = zigzag[1:]
            block_index += 1

    print('Starting entropy code')

    # entropy code
    huffman_dc_Y = HuffmanTree(np.vectorize(bits_required)(dc[:,0]))
    huffman_dc_C = HuffmanTree(np.vectorize(bits_required)(dc[:,1:].flat))
    huffman_ac_Y = HuffmanTree(flatten(RLE(ac[i,:,0])[0]
                                       for i in range(blocks)))
    huffman_ac_C = HuffmanTree(flatten(RLE(ac[i,:,j])[0]
                                       for i in range(blocks) for j in [1,2]))
    tables = {
        'dc_y': huffman_dc_Y.value_to_huffman_repr(),
        'dc_c': huffman_dc_C.value_to_huffman_repr(),
        'ac_y': huffman_ac_Y.value_to_huffman_repr(),
        'ac_c': huffman_ac_C.value_to_huffman_repr(),
    }
    print('Saving to file...')
    save_to_file(OUTPUT_PATH, dc, ac, blocks, tables)

if __name__ == '__main__':
    main()