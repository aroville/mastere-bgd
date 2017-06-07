import sys


class Page:
    def __init__(self, title, content):
        self.content = content
        self.title = title
        if sys.version_info[0] < 3:
            self.title = title.decode("utf-8")
            self.content = content.decode("utf-8")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.title != other.title:
            return False
        return self.content == other.content
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.title, self.content))

    def __str__(self):
        if sys.version_info[0] < 3:
            return 'Wikipedia page: "'+self.title.encode("utf-8")+'"'
        return 'Wikipedia page: "'+self.title+'"'
    
    def __repr__(self):
        return self.__str__()

    def _to_tuple(self):
        return self.title, self.content

    # Only used for Disambiguation TP
    def label(self):
        return self.title[1:self.title.rindex("_")].replace("_", " ")
