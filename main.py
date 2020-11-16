from loader import *
import numpy as np
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.probability import *
from nltk.corpus.reader.wordnet import WordNetError
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression

def baseline(instances, keys):
    cnt = 0
    for entry in instances:
        word = instances[entry].lemma
        m = wn.synsets(word)
        base = wn.synsets(word, instances[entry].pos)[0].lemmas()
        print(wn.synsets(word))

        sense2count = {}  # dictionary of the form {synset_name: count, ...}
        for ss in m:
            freq = 0
            for lemma in ss.lemmas():
                freq += lemma.count()
            sense2count[ss.name()] = freq
            print(freq)
        # print(word," -> " ,m, " -> " ,n,  " -> ",keys[entry])
        list = []
        for b in base:
            list.append((b.key()))
        # print(wn.synsets(word)[0].lemmas())
        # print(list)
        # print(keys[entry])
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
        list = []
        # print(instances[entry].context, "--->",instances[entry].lemma)
        result = (lesk(instances[entry].context, instances[entry].lemma, instances[entry].pos)).lemmas()

        for l in result:
            list.append(l.key())
        print(result)
        print(lesk(instances[entry].context, instances[entry].lemma))
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


def bootstrapping():
    dev_context = []
    dev_lemmas = []
    dev_labels = []
    for entry in dev_instances:
        dev_context.append(dev_instances[entry].lemma+" "+(' '.join(dev_instances[entry].context)))
        dev_lemmas.append(dev_instances[entry].lemma)
        dev_labels.append(str(dev_key[entry]))



    seed_context = []
    seed_lemmas = []
    seed_labels = []
    for entry in dev_instances:
        if (dev_instances[entry].lemma=='year' or dev_instances[entry].lemma=='country' or dev_instances[entry].lemma=='world' or dev_instances[entry].lemma=='week' or dev_instances[entry].lemma=='friday' or dev_instances[entry].lemma=='china' or dev_instances[entry].lemma=='one' or dev_instances[entry].lemma=='deal' or dev_instances[entry].lemma=='impact' or dev_instances[entry].lemma=='united_stated'):
            seed_context.append(dev_instances[entry].lemma+" "+(' '.join(dev_instances[entry].context)))
            seed_lemmas.append(dev_instances[entry].lemma)
            seed_labels.append(str(dev_key[entry]))

    # for entry in test_instances:
    #     if (test_instances[entry].lemma=='year' or test_instances[entry].lemma=='country' or test_instances[entry].lemma=='world' or test_instances[entry].lemma=='week' or test_instances[entry].lemma=='friday' or test_instances[entry].lemma=='china' or test_instances[entry].lemma=='one' or test_instances[entry].lemma=='deal' or test_instances[entry].lemma=='impact' or test_instances[entry].lemma=='united_stated'):
    #         seed_context.append(test_instances[entry].lemma+" "+(' '.join(test_instances[entry].context)))
    #         seed_lemmas.append(test_instances[entry].lemma)
    #         seed_labels.append(str(test_instances[entry]))

    loop=0
    while (loop<5):
        loop=loop+1
        vect = CountVectorizer()
        X=vect.fit_transform(seed_context).toarray()

        dev_X = vect.transform(dev_context).toarray()
        clf=LogisticRegression(max_iter=700)
        clf.fit(X,seed_labels)
        probs=clf.predict_proba(dev_X)
        preds = clf.predict(dev_X)
        for i, prob in enumerate(probs):
            if(max(prob)>0.6):
                seed_context.append(dev_context[i])
                seed_lemmas.append((dev_lemmas[i]))
                seed_labels.append((dev_labels[i]))
        cnt=0;
        for i, pred in enumerate(preds):
            # print(dev_labels[i] , pred)
            if(dev_labels[i] == pred):
                cnt=cnt+1;
        print(cnt/len(preds))

    test_context = []
    test_lemmas = []
    test_labels = []
    for entry in test_instances:
        test_context.append(test_instances[entry].lemma+" "+(' '.join(test_instances[entry].context)))
        test_lemmas.append(test_instances[entry].lemma)
        test_labels.append(str(test_key[entry]))

    test_X = vect.transform(test_context).toarray()
    test_preds = clf.predict(test_X)

    cnt=0
    for i, pred in enumerate(test_preds):
        # print(dev_labels[i] , pred)
        if (test_labels[i] == pred):
            cnt = cnt + 1;
    print("Accuracy on Test",cnt / len(test_preds))

def mixedLesk():

    modi_dev_synsets = {}
    for i in dev_instances.values():
        id = i.id

        base_dev_lemmas = wn.lemmas(i.lemma)
        modi_dev_synsets[id] = base_dev_lemmas[0].key()

        base_dev_prob = [100, 75, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0, 0, 0, 0, 0, 0]
        lesk_dev_lemmas = (lesk(i.context, i.lemma)).lemmas()[0]
        lesk_dev_prob = 10
        max_probability = 0
        range1 = min(len(base_dev_lemmas), 10)

        for x in range(0, range1):
            if (base_dev_prob[x] > max_probability):
                max_probability = base_dev_prob[x]
            if (base_dev_lemmas[x].key() == lesk_dev_lemmas.key()):
                current_prob = base_dev_prob[x] + lesk_dev_prob + len(i.lemma)
                if (current_prob > max_probability):
                    max_probability = current_prob
                    # print(lesk_dev_lemmas.key())
                    modi_dev_synsets[id] = lesk_dev_lemmas.key()

    # Accuracy of Second Modified Algorithm on Development Set
    modi_match_dev = 0;
    for i in dev_key:
        if modi_dev_synsets[i] in dev_key[i]:
            modi_match_dev += 1

    modi_dev_accuracy = modi_match_dev / (float(len(dev_key.values())))

    print("Accuracy of Second Modified Algorithm on Development set: ", modi_dev_accuracy * 100)



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
    #     print(dev_instances[entry].lemma)
    #     print(dev_key[entry])
    # baseline(dev_instances, dev_key)
    # NLTKlesk(dev_instances, dev_key)

    mixedLesk()

    bootstrapping()




