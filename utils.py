from scipy import fftpack
import numpy as np
import os

IMAGE_PATH = './lena.jpg'
OUTPUT_PATH = './encoded.jpg'

Q_y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]])

Q_c = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]])

def DCT2D(block):
    return fftpack.dct(fftpack.dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

def quantize(block, type=1):
    if type == 1:
        return (block / Q_y).round().astype(np.int32)
    elif type == 2:
        return (block / Q_c).round().astype(np.int32)
    else:
        raise ValueError(("type value should be either 1(Q_y) or 2(Q_c)"))

def zigzag_scan(block):
    return np.array([block[point] for point in zigzag_gen(*block.shape)])

# a zigzag generator
def zigzag_gen(row, col):

    def valid(point):
        return 0<=point[0]<row and 0<=point[1]<col

    RIGHT, LEFTDOWN, DOWN, RIGHTUP = range(4)

    map = {
        DOWN: lambda point: (point[0]+1, point[1]),
        RIGHT: lambda point: (point[0], point[1]+1),
        LEFTDOWN: lambda point: (point[0]+1, point[1]-1),
        RIGHTUP: lambda point: (point[0]-1, point[1]+1),
    }

    point = (0,0)
    di = RIGHT

    for _ in range(row*col):
        yield point
        if di == RIGHT or di == DOWN:
            point = map[di](point)
            if valid(map[LEFTDOWN](point)):
                di = LEFTDOWN
            else: di = RIGHTUP
        elif di == RIGHTUP:
            point = map[di](point)
            if not valid(map[di](point)):
                if valid(map[RIGHT](point)):
                    di = RIGHT
                else: di = DOWN
        else:
            point = map[di](point)
            if not valid(map[di](point)):
                if valid(map[DOWN](point)):
                    di = DOWN
                else: di = RIGHT

# get how many bits a number need
def bits_required(num):
    count = 0
    num = abs(num)
    while num > 0:
        num >>= 1
        count += 1
    return count

def bin_repr(num):
    if num == 0: return ''
    # [2:]to remove 0b
    repr = bin(abs(num))[2:]
    if num < 0:
        return ''.join(map(lambda c: '0' if c=='1' else '1', repr))
    return repr

def unit_to_binstr(number, size):
    return bin(number)[2:][-size:].zfill(size)

# run length coding
def RLE(array):
    '''
    :param array:
    :return: (RUNLENGTH, BITLENGTH)
    '''
    last_nonzero = -1
    for i, elem in enumerate(array):
        if elem != 0: last_nonzero = i

    # (runlength, bitlength)
    left_part = []
    # bit reprsent of value
    value = []

    cur_length = 0

    for i, elem in enumerate(array):
        if i > last_nonzero:
            # (0,0) stands for EOB
            left_part.append((0,0))
            value.append(bin_repr(0))
        elif elem == 0 and cur_length < 16:
            cur_length += 1
        else:
            length = bits_required(elem)
            left_part.append((cur_length, length))
            value.append(bin_repr(elem))
    return left_part, value

def flatten(lst):
    return [item for sublst in lst for item in sublst]

def save_to_file(path, dc, ac, blocks_count, tables):
    with open(path, 'w') as f:
        for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
            # 16 bits for table size
            f.write(unit_to_binstr(len(tables[table_name]), 16))

            for key, value in tables[table_name].items():
                if table_name in {'dc_y', 'dc_c'}:
                    f.write(unit_to_binstr(key, 4))
                    f.write(unit_to_binstr(len(value),4))
                    f.write(value)
                else:
                    f.write(unit_to_binstr(key[0], 4))
                    f.write(unit_to_binstr(key[1], 4))
                    f.write(unit_to_binstr(len(value),8))
                    f.write(value)

        # 32 bits for block counts
        f.write(unit_to_binstr(blocks_count, 32))

        for b in range(blocks_count):
            for c in range(3):
                category = bits_required(dc[b,c])
                symbols, values = RLE(ac[b,:,c])

                dc_table = tables['dc_y'] if c==0 else tables['dc_c']
                ac_table = tables['ac_y'] if c==0 else tables['ac_c']

                f.write(dc_table[category])
                f.write(bin_repr(dc[b,c]))

                for i in range(len(symbols)):
                    f.write(ac_table[tuple(symbols[i])])
                    f.write(values[i])




