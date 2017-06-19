res = {}
with open('data/out_max-depth__3.tsv', encoding='utf-8') as f:
    for line in f:
        split = line.split('\t')
        res[split[0]] = split[1]

i = 0
correct = {}
with open('data/goldstandard-sample.tsv', encoding='utf-8') as f:
    for line in f:
        if i == 2000:
            break
        i += 1
        split = line.lower().split('\t')
        correct[split[0]] = split[1]

score = 0
for title, entity in correct.items():
    if title in res and res[title] == entity:
        score += 1

print('score:', score / len(correct))
