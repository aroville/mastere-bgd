import sys
from parser import Parser
from trie import Trie
from multiprocessing import Pool
from time import time


def hunt_in_page(page):
    res = []
    l = len(page.content)
    for s in range(l):
        cl = trie.contained_length(page.content, s)
        if cl > 0:
            res.append(page.content[s:s+cl])
    return '\n'.join(map(lambda e: page.title + '\t' + e, res))

t0 = time()
trie = Trie()
trie.load(sys.argv[1])
with Pool() as p:
    results = p.map(hunt_in_page, Parser(sys.argv[2]))
with open(sys.argv[3], 'w+') as f:
    f.write('\n'.join([r for r in results if r]))

print('Finished, ellapsed: {:.2f}s'.format(time()-t0))
