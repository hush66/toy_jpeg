from PIL import Image
from utils import *

def decoder():
    print('Start decode...')
    dc, ac, tables, blocks_number, = read_image_file(OUTPUT_PATH)
    print('File readed.')
    block_pre_line = int(math.sqrt(blocks_number))
    image_side = block_pre_line * BLOCK_SIDE

    matrix = np.empty((image_side, image_side, 3), dtype=np.uint8)

    for block_index in range(blocks_number):
        i = block_index // block_pre_line * BLOCK_SIDE
        j = block_index % block_pre_line * BLOCK_SIDE

        for c in range(3):
            rezigzaged = zigzag_to_block([dc[block_index, c]] + list(ac[block_index, :, c]))
            dequantized = dequantize(rezigzaged, 1 if c == 0 else 2)
            idcted = iDCT2D(dequantized)
            matrix[i:i+8, j:j+8, c] = idcted + 128

    image = Image.fromarray(matrix, 'YCbCr')
    image = image.convert('RGB')
    image.save(REBUILD_PATH)
    print('Image rebuilded.')