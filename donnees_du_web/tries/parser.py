from page import Page


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
