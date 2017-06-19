import sys
from parser import Parser
from multiprocessing import Pool
from simpleKB import SimpleKB
from collections import defaultdict
from operator import itemgetter
from nltk.corpus import stopwords

max_depth = 1

if len(sys.argv) == 1:
    sys.argv.append('data/yagoLinks.tsv')
    sys.argv.append('data/yagoLabels.tsv')
    sys.argv.append('data/wikipedia-ambiguous.txt')
    sys.argv.append('data/out_max-depth__{}.tsv'.format(max_depth))

sw = stopwords.words('english')
yago = SimpleKB(sys.argv[1], sys.argv[2])
print('Loading', sys.argv[1], end='...', flush=True)
pages = list(Parser(sys.argv[3]))
print("done", flush=True)


def find_closest_entity(page):
    starting_entities = set(yago.entities_by_label(page.label().lower()))
    content = page.content.lower()

    scores = defaultdict(int)
    for e in starting_entities:
        cur = [e]
        seen = set()

        for i in range(max_depth, 0, -1):
            cur.extend(yago.linked_entities(cur))
            cur = [e for e in cur if e not in seen]

            if len(cur) == 0:
                break

            # If we recognize a label in the page content, increment the score
            for label in yago.labels_by_entities(cur):
                if label in content:  # and label not in sw:
                    scores[e] += i

            # Update the list of covered entities
            seen = seen.union(cur)

    if len(scores) == 0:
        return page.title, list(starting_entities)[0]
    return page.title, max(scores.items(), key=itemgetter(1))[0]


nb_pages = len(pages)
with Pool() as p:
    res = p.map(find_closest_entity, pages)
with open(sys.argv[4], 'w', encoding='utf-8') as f:
    f.write('\n'.join([r[0]+'\t'+r[1] for r in res if r]))


res = {}
with open(sys.argv[4], encoding='utf-8') as f:
    for line in f:
        split = line.strip('\n').split('\t')
        res[split[0]] = split[1]

i = 0
correct = {}
with open('data/goldstandard-sample.tsv', encoding='utf-8') as f:
    for line in f:
        if i == 2000:
            break
        i += 1
        split = line.strip('\n').lower().split('\t')
        correct[split[0]] = split[1]

score = 0
for t, e in correct.items():
    if res.get(t) == e:
        score += 1

print('\nscore: {:.3f}%'.format(100*score / len(correct)))
