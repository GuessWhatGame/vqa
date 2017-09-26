import io
import json
import collections

from vqa.data_provider.vqa_dataset import VQADataset
from nltk.tokenize import TweetTokenizer

if __name__ == '__main__':

    min_nr_of_occurrences = 2
    keep_top_answers = 1000

    preprocess_answers = True

    data_dir = '/home/sequel/fstrub/vqa_data'
    data_dir = '/data/lisa/data/vqa'
    year = '2014'

    train_dataset = VQADataset(data_dir, year, "train", preprocess_answers=preprocess_answers)
    valid_dataset = VQADataset(data_dir, year, "val", preprocess_answers=preprocess_answers)

    answer_counters = train_dataset.answer_counter.most_common() + valid_dataset.answer_counter.most_common()
    games = train_dataset.games + valid_dataset.games

    word2i = {'<unk>': 0,
              '<start>': 1,
              '<stop>': 2,
              '<padding>': 3
              }

    answer2i = {'<unk>': 0}

    word2occ = collections.defaultdict(int)
    answer2occ = collections.Counter()


    for k, v in answer_counters:
        answer2occ[k] += v

    selected = sum([v[1] for v in answer2occ.most_common(keep_top_answers)])
    total = sum([v[1] for v in answer2occ.most_common()])
    print(float(selected)/total)

    # Input words
    tknzr = TweetTokenizer(preserve_case=False)

    for game in games:
        input_tokens = tknzr.tokenize(game.question)
        for tok in input_tokens:
            word2occ[tok] += 1


    included_cnt = 0
    excluded_cnt = 0
    for word, occ in word2occ.items():
        if occ >= min_nr_of_occurrences and word.count('.') <= 1:
            included_cnt += occ
            word2i[word] = len(word2i)
        else:
            excluded_cnt += occ


    for i, answer in enumerate(answer2occ.most_common(keep_top_answers)):
        answer2i[answer[0]] = len(answer2i)

    print(len(word2i))
    print(len(answer2i))


    save_file = data_dir + "/dict_vqa_"+ str(year) +"_" + str(keep_top_answers) + "answers2.json"
    with io.open(save_file, 'w', encoding='utf8') as f_out:
       data = json.dumps({'word2i': word2i, 'answer2i': answer2i, "preprocess_answers": preprocess_answers})
       f_out.write(data)
       #f_out.write(unicode(data))



