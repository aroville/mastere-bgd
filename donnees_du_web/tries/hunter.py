"""Given as command line arguments
  (1) a file with a list of entities, 
  (2) a Wikipedia file, and
  (3) an output file
  writes to the output file lines of the form
        title TAB entity
  where <title> is the title of a Wikipedia article,
  and <entity> is an entity from the entity file
  that occurs in the text of that article. 
  All occurrences of entities at all positions of
  the article text shall be output, including duplicates. 
  (Public skeleton code)"""
  
import sys
from parser import Parser
from trie import Trie
import math
from multiprocessing import Pool

wiki_file = 'wikipedia-first/wikipedia-first.txt'

if __name__ == '__main__':
    entities = sys.argv[1] if len(sys.argv) >= 2 else 'entities.txt'
    wiki = sys.argv[2] if len(sys.argv) >= 3 else wiki_file
    out = sys.argv[3] if len(sys.argv) >= 4 else 'out.txt'

    trie = Trie()
    trie.load('entities.txt')

    parser = Parser(wikipedia_file=wiki_file)

    def hunt_in_page(page):
        res = []
        title = page.title
        content = page.content

        while len(content) > 0:
            cl = trie.contained_length(content, 0)
            if math.isnan(cl) or cl == 0:
                content = content[1:]
                continue

            res.append('{}\t{}'.format(title, content[:cl]))
            content = content[cl:]
        return '\n'.join(res)

    with Pool() as p:
        results = p.map(hunt_in_page, parser)

    with open(out, 'w+') as f:
        f.write('\n'.join(results))
