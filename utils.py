from scipy import fftpack
import numpy as np
import math
import os

# Constant Value
IMAGE_PATH = './lena.jpg'
OUTPUT_PATH = './encoded'
REBUILD_PATH = './rebuild.jpg'
BLOCK_SIDE = 8


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


######################
# DCT and iDCT
######################
def DCT2D(block):
    return fftpack.dct(fftpack.dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

def iDCT2D(block):
    return fftpack.idct(fftpack.idct(block.T, norm='ortho').T, norm='ortho')

###########################
# Quantize and dequantize
###########################
def quantize(block, type=1):
    if type == 1:
        return (block / Q_y).round().astype(np.int32)
    elif type == 2:
        return (block / Q_c).round().astype(np.int32)
    else:
        raise ValueError(("type value should be either 1(Q_y) or 2(Q_c)"))

def dequantize(block, type=1):
    if type == 1:
        return block * Q_y
    elif type == 2:
        return block * Q_c
    else:
        raise ValueError(("type value should be either 1(Q_y) or 2(Q_c)"))

######################
# zigzag and rezigzag
######################
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

def zigzag_to_block(zigzag):
    row = col = int(math.sqrt(len(zigzag)))
    block = np.empty((row, col), np.int32)
    for i, position in enumerate(zigzag_gen(row,col)):
        block[position] = zigzag[i]
    return block

######################
# number operation
######################
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

def binstr_flip(binstr):
    return ''.join(map(lambda c: '0' if c=='1' else '1', binstr))

######################
# Run Length coding
######################
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
            break
        elif elem == 0 and cur_length < 15:
            cur_length += 1
        else:
            length = bits_required(elem)
            left_part.append((cur_length, length))
            value.append(bin_repr(elem))
            cur_length = 0
    return left_part, value

def flatten(lst):
    return [item for sublst in lst for item in sublst]

######################
# File write and load
######################
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


class JPEGFileReader:
    TABLE_SIZE_BITS = 16
    BLOCKS_COUNT_BITS = 32

    DC_CODE_LENGTH_BITS = 4
    CATEGORY_BITS = 4

    AC_CODE_LENGTH_BITS = 8
    RUN_LENGTH_BITS = 4
    SIZE_BITS = 4

    def __init__(self, filepath):
        self.__file = open(filepath, 'r')

    def read_int(self, size):
        if size == 0:
            return 0
        # get the right represent
        bin_num = self.__read_str(size)
        if bin_num[0] == '1':
            return self.__int2(bin_num)
        else:
            return self.__int2(binstr_flip(bin_num)) * -1

    def read_dc_table(self):
        table = dict()

        table_size = self.__read_uint(self.TABLE_SIZE_BITS)
        for _ in range(table_size):
            category = self.__read_uint(self.CATEGORY_BITS)
            code_length = self.__read_uint(self.DC_CODE_LENGTH_BITS)
            code = self.__read_str(code_length)
            table[code] = category
        return table

    def read_ac_table(self):
        table = dict()

        table_size = self.__read_uint(self.TABLE_SIZE_BITS)
        for _ in range(table_size):
            run_length = self.__read_uint(self.RUN_LENGTH_BITS)
            size = self.__read_uint(self.SIZE_BITS)
            code_length = self.__read_uint(self.AC_CODE_LENGTH_BITS)
            code = self.__read_str(code_length)
            table[code] = (run_length, size)
        return table

    def read_blocks_count(self):
        return self.__read_uint(self.BLOCKS_COUNT_BITS)

    def read_huffman_code(self, table):
        prefix = ''
        while prefix not in table:
            prefix += self.__read_char()
        return table[prefix]

    def __read_uint(self, size):
        if size <= 0:
            raise ValueError("size of unsigned int should be greater than 0")
        return self.__int2(self.__read_str(size))

    def __read_str(self, length):
        return self.__file.read(length)

    def __read_char(self):
        return self.__read_str(1)

    def __int2(self, bin_num):
        return int(bin_num, 2)


def read_image_file(filepath):
    reader = JPEGFileReader(filepath)

    tables = dict()
    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
        if 'dc' in table_name:
            tables[table_name] = reader.read_dc_table()
        else:
            tables[table_name] = reader.read_ac_table()

    blocks_count = reader.read_blocks_count()

    dc = np.empty((blocks_count, 3), dtype=np.int32)
    ac = np.empty((blocks_count, 63, 3), dtype=np.int32)

    for block_index in range(blocks_count):
        for component in range(3):
            dc_table = tables['dc_y'] if component == 0 else tables['dc_c']
            ac_table = tables['ac_y'] if component == 0 else tables['ac_c']

            category = reader.read_huffman_code(dc_table)
            dc[block_index, component] = reader.read_int(category)

            cells_count = 0

            while cells_count < 63:
                run_length, size = reader.read_huffman_code(ac_table)

                if (run_length, size) == (0, 0):
                    while cells_count < 63:
                        ac[block_index, cells_count, component] = 0
                        cells_count += 1
                else:
                    for i in range(run_length):
                        ac[block_index, cells_count, component] = 0
                        cells_count += 1
                    if size == 0:
                        ac[block_index, cells_count, component] = 0
                    else:
                        value = reader.read_int(size)
                        ac[block_index, cells_count, component] = value
                    cells_count += 1

    return dc, ac, tables, blocks_count