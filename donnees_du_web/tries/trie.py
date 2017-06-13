import math
from collections import defaultdict


class Trie:
    def __init__(self):
        self.checked = False
        self.d = defaultdict(Trie)
        
    def load(self, file):
        with open(file, encoding="utf-8") as entities:
            for entity in entities:                
                self.add(entity.strip())
        
    def add(self, word, start=0):
        if len(word) == start:
            self.checked = True
        else:
            self.d[word[start]].add(word, start+1)
    
    def contained_length(self, word, start):
        if start < len(word) and word[start] in self.d:
            l = self.d[word[start]].contained_length(word, start+1)
            if not math.isnan(l):
                return 1 + l
        return 0 if self.checked else float('nan')
