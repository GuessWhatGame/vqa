import os
import pickle

import guesswhat.data_provider as provider
from guesswhat.data_provider.nlp_preprocessors import VQATokenizer, GloveEmbeddings

data_dir = '/home/sequel/fstrub/vqa_data'
year = 2014

trainset = provider.VQADataset(data_dir, year=year, which_set="train")
validset = provider.VQADataset(data_dir, year=year, which_set="val")
testset = provider.VQATestDataset(data_dir, year=year, which_set="test-dev")

tokenizer = VQATokenizer(os.path.join(data_dir, 'dict_vqa_2014.json'))


# wget http://nlp.stanford.edu/data/glove.42B.300d.zip

vectors_file = '/home/sequel/hdevries/glove.42B.300d.txt'
with open(vectors_file, 'r') as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

glove_dict = {}
not_in_dict = {}
for set in [trainset, validset, testset]:
    for g in set.games:
        words = tokenizer.tokenize_question(g.question)
        for w in words:
            w = w.lower()
            if w in vectors:
                glove_dict[w] = vectors[w]
            else:
                not_in_dict[w] = 1

print(len(glove_dict))
print(len(not_in_dict))

for k in not_in_dict.keys():
    print(k)

pickle.dump(glove_dict, open('glove_dict.pkl', 'wb'), protocol=2)





