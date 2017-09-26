import collections
import numpy as np

from generic.data_provider.nlp_utils import padder, padder_3d
from generic.data_provider.batchifier import AbstractBatchifier


class VQABatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, sources, glove=None, remove_unknown=False, answer_type=None):
        self.tokenizer = tokenizer
        self.remove_unknown = remove_unknown
        self.answer_type = answer_type
        self.sources = sources
        self.glove = glove

        # should be moved somewehere else
        assert 'glove' not in sources or glove is not None


    def filter(self, games):

        if self.remove_unknown:
            games = [g for g in games if self.tokenizer.encode_answer(g.majority_answer) !=  self.tokenizer.unknown_answer]

        if self.answer_type is not None:
            games = [g for g in games if g.answer_type == self.answer_type]

        return games

    def apply(self, games):

        batch = collections.defaultdict(list)
        batch_size = len(games)

        assert batch_size > 0

        for i, game in enumerate(games):

            batch["raw"].append(game)

            # Get question
            question = self.tokenizer.encode_question(game.question)
            batch['question'].append(question)

            if 'glove' in self.sources:
                # Add glove vectors
                words = self.tokenizer.tokenize_question(game.question)
                glove_vectors = self.glove.get_embeddings(words)
                batch['glove'].append(glove_vectors)

            # Get answers
            if  "answer_count" not in batch: # initialize an empty array for better memory consumption
                batch["answer_count"] = np.zeros((batch_size,self.tokenizer.no_answers))

            for answer in game.answers:
                answer_id = self.tokenizer.encode_answer(answer)
                if answer_id == self.tokenizer.unknown_answer and self.remove_unknown:
                    continue
                batch["answer_count"][i][answer_id] += 1

            # retrieve the image source type
            img = game.picture.get_image()
            if "picture" not in batch: # initialize an empty array for better memory consumption
                batch["picture"] = np.zeros((batch_size,) + img.shape)
            batch["picture"][i] = img

        # pad the questions
        batch['question'], batch['seq_length'] = padder(batch['question'],
                                                        padding_symbol=self.tokenizer.padding_token)
        if 'glove' in self.sources:
            batch['glove'], _ = padder_3d(batch['glove'])
	
        # create mask
        max_len = batch['seq_length'].max()
        batch['seq_mask'] = np.zeros((batch_size, max_len))
        for i, l in enumerate(batch['seq_length']):
            batch['seq_mask'][i, :l] = 1.0

        return batch
