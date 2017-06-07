import sys
from parser import Parser
from trie import Trie
import math
from multiprocessing import Pool


def hunt_in_page(page):
    res = []
    content = page.content
    while len(content) > 0:
        cl = trie.contained_length(content, 0)
        if math.isnan(cl) or cl == 0:
            content = content[1:]
            continue
        res.append('{}\t{}'.format(page.title, content[:cl]))
        content = content[cl:]
    return '\n'.join(res)

trie = Trie()
trie.load(sys.argv[1])
with Pool() as p:
    results = p.map(hunt_in_page, Parser(sys.argv[2]))

results = [r for r in results if r]
with open(sys.argv[3], 'w+') as f:
    f.write('\n'.join(results))
