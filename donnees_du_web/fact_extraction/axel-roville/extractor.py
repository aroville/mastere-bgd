import sys
from parser import Parser
from multiprocessing import Pool, Manager
from nltk import word_tokenize, pos_tag

vague = ['kind', 'sort', 'type']
separators = ['is', 'was', 'are', 'were', 'mean', 'means', 'refers', 'reigned',
              'refers', 'refer', 'gave the designation']

if len(sys.argv) == 1:
    sys.argv.append('wikipedia-first.txt')
    sys.argv.append('output.txt')


def extract_type(page):
    title = page.title.lower()
    tokens = word_tokenize(page.content.lower())
    tags = pos_tag(tokens)

    for i in range(len(tokens)):
        if tokens[i] in separators:
            tags = tags[i+1:]
            break

    last_found_noun = None
    for w, tag in tags:
        if tag.startswith('NN') and w not in vague and not title.startswith(w):
            last_found_noun = w
        elif last_found_noun is not None:
            results.append(page.title+'\t'+last_found_noun)
            return

results = Manager().list()
with Pool() as p:
    p.map(extract_type, list(Parser(sys.argv[1])))
with open(sys.argv[2], 'w') as output:
    output.write('\n'.join(results))
