from loader import *
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.probability import *


def baseline(instances, keys):
    cnt = 0
    for entry in instances:
        word = instances[entry].lemma
        m = wn.synsets(word)[0]
        base = wn.synsets(word)[0].lemmas()
        # print(word," -> " ,m, " -> " ,n,  " -> ",keys[entry])
        list = []
        for b in base:
            list.append((b.key()))
        for key in keys[entry]:
            # print (keys[entry],"---",key, "---",list)
            if key in list:
                cnt += 1
    baseline_accuracy = float(cnt) / len(instances)
    print(" Baseline: ", baseline_accuracy * 100)





def NLTKlesk(instances, keys):
    sum = 0
    count = len(instances)


    for entry in instances:
        list= []
        # print(instances[entry].context, "--->",instances[entry].lemma)
        result = (lesk(instances[entry].context, instances[entry].lemma)).lemmas()

        for l in result:
            list.append(l.key())
        print(result)
        print(list)
        print(keys[entry])
        for key in keys[entry]:
            if key in list:
                sum += 1
    lesk_accuracy = float(sum) / len(instances)
    print("Lesk accuracy: " + str(lesk_accuracy * 100) + "%")


def preprocess(instances):
    for key, instance in instances.items():
        instance.lemma = instance.lemma.decode("utf-8").lower()
        post = []
        # print(instance.lemma)
        stops = set(stopwords.words('english'))
        # print(instance.context)
        for token in instance.context:

            if len(token) > 1 and token not in stops:
                post.append(token.decode("utf-8").lower().replace('@', ''))
        instance.context = post
        # print(instance.context)


if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'

    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)
    preprocess(dev_instances)
    preprocess(test_instances)
    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for (k, v) in dev_instances.items() if k in dev_key}
    test_instances = {k: v for (k, v) in test_instances.items() if k in test_key}
    # for entry in dev_instances:
    #     print(dev_instances[entry].context)
    baseline(test_instances, test_key)
    NLTKlesk(test_instances, test_key)
