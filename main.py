from loader import *
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn


def baseline(instances, keys):
    cnt = 0
    for entry in instances:
        word = instances[entry].lemma.decode("utf-8")
        m = wn.synsets(word)[0]
        base = wn.synsets(word)[0].lemmas()
        # print(word," -> " ,m, " -> " ,n,  " -> ",keys[entry])
        list = []
        for b in base:
            list.append((b.key()))
        for key in keys[entry]:
            print (keys[entry],"---",key, "---",list)
            if key in list:
                cnt += 1
    baseline_accuracy = float(cnt) / len(instances)
    print(" Baseline: " ,baseline_accuracy * 100)

if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'

    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k:v for (k,v) in dev_instances.items() if k in dev_key}
    test_instances = {k:v for (k,v) in test_instances.items() if k in test_key}

    baseline(test_instances, test_key)