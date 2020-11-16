'''
@author: jcheung

Developed for Python 2. Automatically converted to Python 3; may result in bugs.
'''
import xml.etree.cElementTree as ET
import codecs
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

class WSDInstance:
    def __init__(self, my_id, lemma, context, index, pos=None):
        self.id = my_id         # id of the WSD instance
        self.lemma = lemma      # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.index = index      # index of lemma within the context
        self.pos = pos
    def __str__(self):
        '''
        For printing purposes.
        '''
        return '%s\t%s\t%s\t%d' % (self.id, self.lemma, ' '.join(self.context), self.index)
    # def __repr__(self):
    #     return '%s\t%s\t%s\t%d' % (self.id, self.lemma, ' '.join(self.context), self.index)

def load_instances(f):
    '''
    Load two lists of cases to perform WSD on. The structure that is returned is a dict, where
    the keys are the ids, and the values are instances of WSDInstance.
    '''
    tree = ET.parse(f)
    root = tree.getroot()
    
    dev_instances = {}
    test_instances = {}
    
    for text in root:
        if text.attrib['id'].startswith('d001'):
            for sentence in text:
                # construct sentence context
                context = [to_ascii(el.attrib['lemma']) for el in sentence]
                # print(context)
                for i, el in enumerate(sentence):
                    if el.tag == 'instance':
                        my_id = el.attrib['id']
                        lemma = to_ascii(el.attrib['lemma'])
                        pos = el.attrib['pos'][0].lower()
                        dev_instances[my_id] = WSDInstance(my_id, lemma, context, i)
                        # print(dev_instances.values())


        else:
            for sentence in text:
                # construct sentence context
                context = [to_ascii(el.attrib['lemma']) for el in sentence]
                # print(context)
                for i, el in enumerate(sentence):
                    if el.tag == 'instance':
                        my_id = el.attrib['id']
                        lemma = to_ascii(el.attrib['lemma'])
                        pos = el.attrib['pos'][0].lower()
                        test_instances[my_id] = WSDInstance(my_id, lemma, context, i)
                        # print(my_id)



    return dev_instances, test_instances

def load_key(f):
    '''
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys. 
    '''
    dev_key = {}
    test_key = {}
    for line in open(f):
        if len(line) <= 1: continue
        #print (line)
        doc, my_id, sense_key = line.strip().split(' ', 2)
        if doc == 'd001':

            dev_key[my_id] = sense_key.split()
        else:
            test_key[my_id] = sense_key.split()
    return dev_key, test_key

def to_ascii(s):
    # remove all non-ascii characters
    return codecs.encode(s, 'ascii', 'ignore')

if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k:v for (k,v) in dev_instances.items() if k in dev_key}
    test_instances = {k:v for (k,v) in test_instances.items() if k in test_key}
    
    # read to use here
    print(len(dev_instances)) # number of dev instances
    print(list(dev_instances.keys()))

    print(test_instances)
    print(len(test_instances)) # number of test instances

    sent = ['I', 'went', 'to', 'the', 'bank', 'to', 'deposit', 'money', '.']
    print(lesk(sent, 'bank', 'v'))
    print(lesk(sent, 'bank'))
    print(wn.synsets('bank'))
    print(type(dev_instances))


