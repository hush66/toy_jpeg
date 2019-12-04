from queue import PriorityQueue

class HuffmanTree:

    class __Node:
        def __init__(self, freq, value, left, right):
            self.freq, self.value, self.left, self.right = freq, value, left, right

        def __lt__(self, other):
            return self.freq < other.freq

    def __init__(self, array):
        q = PriorityQueue()

        for val, freq in self.__frequency(array).items():
            q.put(self.__Node(freq, val, None, None))

        self.__buildTree(q)
        self.__root = q.get()

        self.__huffman_repr = dict()

    def __frequency(self, array):
        frequency = dict()
        for num in array:
            if num in frequency.keys():
                frequency[num] += 1
            else: frequency[num] = 1
        return frequency

    def __buildTree(self, pq):
        while pq.qsize() >= 2:
            tmp1 = pq.get()
            tmp2 = pq.get()
            newNode = self.__Node(tmp1.freq+tmp2.freq, None, tmp1, tmp2)
            pq.put(newNode)

    def value_to_huffman_repr(self):
        if len(self.__huffman_repr.keys()) == 0:
            self.build_huffman_table()
        return self.__huffman_repr

    def build_huffman_table(self):
        def traverse(node, huffmancode=''):
            if not node: return
            elif not node.left and not node.right:
                self.__huffman_repr[node.value] = huffmancode
                return
            traverse(node.left, huffmancode+'0')
            traverse(node.right, huffmancode+'1')
        traverse(self.__root)