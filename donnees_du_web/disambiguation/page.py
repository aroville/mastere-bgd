import sys


class Page:
    def __init__(self, title, content):
        self.content = content
        self.title = title

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
               and self.title == other.title \
               and self.content == other.content
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.title, self.content))

    def __str__(self):
        return 'Wikipedia page: "'+self.title+'"'
    
    def __repr__(self):
        return self.__str__()

    def _to_tuple(self):
        return self.title, self.content

    # Only used for Disambiguation TP
    def label(self):
        return self.title[1:self.title.rindex('_')].replace('_', ' ')
