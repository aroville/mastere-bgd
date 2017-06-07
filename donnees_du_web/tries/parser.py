"""Parses a Wikipedia file, returns page objects"""

from page import Page
__author__ = "Jonathan Lajus"


class Parser:
    def __init__(self, wikipedia_file):
        self.file = wikipedia_file

    def __iter__(self):
        title, content = None, ""
        with open(self.file, encoding='utf-8') as file:
            for line in file:                
                line = line.strip()
                if not line and title is not None:
                    yield Page(title, content.rstrip())
                    title, content = None, ""
                elif title is None:
                    title = line
                else:
                    content += line + " "

if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    else:
        filename = 'wikipedia-first/wikipedia-first.txt'

    parser = Parser(filename)
    for page in parser:
        print(page)
