import math
from collections import defaultdict
"""This module implements a Trie (Public skeleton code)"""


class Trie:
    """Represents a Trie. Should not store the strings in a 
    parallel data structure (such as a map or a list)"""

    def __init__(self):
        self.checked = False
        self.d = defaultdict(Trie)
        
    def load(self, file):
        """Load all lines of the file as elements of the trie""" 
        print("Loading", file, end="...", flush=True)
        with open(file, encoding="utf-8") as entities:
            for entity in entities:                
                self.add(entity.strip())
        print("done", flush=True)
        
    def add(self, word, start=0):
        """Add the string given by word[start:]"""
        if len(word) == start:
            self.checked = True
        else:
            self.d[word[start]].add(word, start+1)
    
    def contained_length(self, word, start):
        """Return the largest i such that word[start:start+i] is in the Trie"""
        if start < len(word) and word[start] in self.d:
            l = self.d[word[start]].contained_length(word, start+1)
            if not math.isnan(l):
                return 1 + l
        return 0 if self.checked else float('nan')
