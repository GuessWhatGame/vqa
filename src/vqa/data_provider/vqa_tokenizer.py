from nltk.tokenize import TweetTokenizer
import json
import re
import os


class VQATokenizer:
    """ """
    def __init__(self, dictionary_file, use_mcb_tokenizer=False):

        if use_mcb_tokenizer:
            self.tokenizer_fn = seq_to_list
            data_dir = os.path.dirname(dictionary_file)

            vdict_path = os.path.join(data_dir, 'vdict.json')
            with open(vdict_path) as f:
                self.word2i = json.load(f)
                self.word2i['<unk>'] = self.word2i['']

            adict_path = os.path.join(data_dir, 'adict.json')
            with open(adict_path) as f:
                self.answer2i = json.load(f)
                self.answer2i['<unk>'] = self.answer2i['']

        else:
            self.tokenizer_fn = TweetTokenizer(preserve_case=False).tokenize
            with open(dictionary_file, 'r') as f:
                data = json.load(f)
                self.word2i = data['word2i']
                self.answer2i = data['answer2i']

        self.dictionary_file = dictionary_file

        self.i2word = {}
        for (k, v) in self.word2i.items():
            self.i2word[v] = k

        self.i2answer = {}
        for (k, v) in self.answer2i.items():
            self.i2answer[v] = k

        # Retrieve key values
        self.no_words = len(self.word2i)
        self.no_answers = len(self.answer2i)

        self.unknown_question_token = self.word2i["<unk>"]
        self.padding_token = self.word2i["<unk>"]

        self.unknown_answer = self.answer2i["<unk>"]


    """
    Input: String
    Output: List of tokens
    """
    def encode_question(self, question):
        tokens = []
        for token in self.tokenizer_fn(question):
            if token not in self.word2i:
                token = '<unk>'
            tokens.append(self.word2i[token])
        return tokens

    def decode_question(self, tokens):
        return ' '.join([self.i2word[tok] for tok in tokens])

    def encode_answer(self, answer):
        if answer not in self.answer2i:
            return self.answer2i['<unk>']
        return self.answer2i[answer]

    def decode_answer(self, answer_id):
        return self.i2answer[answer_id]

    def tokenize_question(self, question):
        return self.tokenizer_fn(question)


def seq_to_list(s):
    t_str = s.lower()
    for i in [r'\?', r'\!', r'\'', r'\"', r'\$', r'\:', r'\@', r'\(', r'\)', r'\,', r'\.', r'\;']:
        t_str = re.sub(i, '', t_str)
    for i in [r'\-', r'\/']:
        t_str = re.sub(i, ' ', t_str)
    q_list = re.sub(r'\?', '', t_str.lower()).split(' ')
    return q_list




