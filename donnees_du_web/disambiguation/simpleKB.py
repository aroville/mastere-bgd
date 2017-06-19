def load(file_name, container, reverse_container):
    with open(file_name, encoding="utf-8") as file:
        print("Loading", file_name, end="...", flush=True)
        for line in file:
            split_line = line.lower().split('\t')
            subject = split_line[0].strip()
            obj = split_line[1].strip('"\n').strip()
            container.setdefault(subject, set()).add(obj)
            reverse_container.setdefault(obj, set()).add(subject)
        print("done", flush=True)


class SimpleKB:
    def __init__(self, yago_links_file, yago_labels_file):
        self.links = {}
        self.labels = {}
        self.rlabels = {}
        load(yago_links_file, self.links, self.links)
        load(yago_labels_file, self.labels, self.rlabels)

    def entities_by_label(self, label):
        if label in self.rlabels:
            return self.rlabels[label]
        return set()

    def labels_by_entity(self, entity):
        if entity in self.labels:
            return self.labels[entity]
        return set()

    def linked_entities(self, entities):
        res = []
        for entity in entities:
            res.extend(self.links[entity])
        return set(res)

    def entities_by_labels(self, labels):
        res = []
        for label in labels:
            res.extend(self.entities_by_label(label))
        return set(res)

    def labels_by_entities(self, entities):
        res = []
        for entity in entities:
            res.extend(self.labels_by_entity(entity))
        return set(res)
