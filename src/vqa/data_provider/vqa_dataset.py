import collections
import json
import os

from generic.data_provider.dataset import AbstractDataset
from vqa.preprocess_data.vqa_nlp_processor import VQAAnswerProcessor

use_100 = False


class Picture:
    def __init__(self, id, image_loader):
        self.id = id
        self.url = "http://mscoco.org/images/{}".format(id)
        self.filename = str(id).zfill(12) + ".jpg"

        if image_loader is not None:
            self.image_loader = image_loader.preload(id)
            self.file = os.path.join(image_loader.img_dir, self.filename)

    def get_image(self):
        return self.image_loader.get_image(self.file)


class Game(object):
    def __init__(self, id, picture, question, majority_answer, answers, question_type, answer_type):
        self.id = id
        self.picture = picture
        self.question = question
        self.majority_answer = majority_answer
        self.answers = answers
        self.question_type = question_type
        self.answer_type = answer_type

    def __str__(self):
        return "[#q:{}, #p:{}] {} - {} ({})".format(self.id, self.picture.id, self.question, self.majority_answer, self.answer_type)


class VQADataset(AbstractDataset):
    """Loads the dataset."""

    def __init__(self, folder, year, which_set, image_loader=None, preprocess_answers=False):

        annotations_path_file = '{}/vqa_{}{}_annotations.json'.format(folder, which_set, year)
        questions_path_file = '{}/vqa_{}{}_questions.json'.format(folder, which_set, year)

        games = []
        self.question_types = collections.Counter()

        self.answer_counter = collections.Counter()
        self.answer_types = collections.Counter()

        vqa_preprocessor = VQAAnswerProcessor()

        with open(annotations_path_file) as annotations_file:
            with open(questions_path_file) as questions_file:
                print("Loading annotations...")
                full_annotations = json.load(annotations_file)

                print("Loading questions...")
                full_questions = json.load(questions_file)

                assert full_annotations["info"]["version"] == full_questions["info"]["version"]
                assert full_annotations["data_subtype"] == full_questions["data_subtype"]
                assert full_annotations["data_subtype"].startswith(which_set)
                assert full_annotations["data_subtype"].endswith(str(year))

                print("Successfully Loaded VQA v{} ({})".format(full_annotations["info"]["version"], which_set))

                for annotation, question in zip(full_annotations["annotations"], full_questions["questions"]):
                    assert annotation["question_id"] == question["question_id"]

                    question_id = int(question["question_id"])
                    picture_id = question["image_id"]

                    question = question["question"]
                    question_type = annotation["question_type"]
                    self.question_types[question_type] += 1

                    majority_answer = annotation["multiple_choice_answer"]
                    answers = [ a["answer"] for a in annotation["answers"]]
                    answer_type = annotation["answer_type"]

                    if preprocess_answers:
                        majority_answer = vqa_preprocessor.preprocess_answers(majority_answer)
                        answers = [vqa_preprocessor.preprocess_answers(a) for a in answers]

                    for a in answers:
                        self.answer_counter[a] += 1
                    self.answer_types[answer_type] += 1

                    games.append(Game(id=question_id,
                                           picture=Picture(picture_id, image_loader),
                                           question=question,
                                           question_type=question_type,
                                           majority_answer=majority_answer,
                                           answers=answers,
                                           answer_type=answer_type))

                    if use_100 and len(games) > 100: break

        print('{} games loaded...'.format(len(games)))
        super(VQADataset, self).__init__(games)

#TODO split annotation loading to question loading (code duplication)
class VQATestDataset(AbstractDataset):
    """Loads the dataset."""

    def __init__(self, folder, year, which_set, image_loader=None):

        questions_path_file = '{}/vqa_{}{}_questions.json'.format(folder, which_set, year)

        games = []
        self.question_types = collections.Counter()

        with open(questions_path_file) as questions_file:

            print("Loading questions...")
            full_questions = json.load(questions_file)

            assert full_questions["data_subtype"] == full_questions["data_subtype"]
            assert full_questions["data_subtype"].startswith(which_set)

            print("Successfully Loaded VQA v{} ({})".format(full_questions["info"]["version"], which_set))

            for question in full_questions["questions"]:

                question_id = int(question["question_id"])
                picture_id = question["image_id"]
                question = question["question"]

                games.append(Game(id=question_id,
                                       picture=Picture(picture_id, image_loader),
                                       question=question,
                                       question_type="All",
                                       majority_answer="<unk>",
                                       answers=["<unk>"],
                                       answer_type="All"))

                if use_100 and len(games) > 100: break

        print('{} games loaded...'.format(len(games)))
        super(VQATestDataset, self).__init__(games)


# class VQATestWriter(object):
#     def __init__(self, tokenizer, file_path):
#         self.file_path = file_path
#         self.tokenizer = tokenizer
#         self.results = []
#
#     def append(self, results, batch):
#         for answer_id, sample in zip(results, batch['raw']):
#             self.results.append({'question_id': sample.id,
#                                  'answer': self.tokenizer.decode_answer(answer_id)})
#
#     def dump(self):
#         dump_json(self.file_path, self.results)


if __name__ == '__main__':
    dataset = VQADataset("/home/sequel/fstrub/vqa_data", year=2014, which_set="val")
    print(dataset.question_types)
    print(dataset.question_types)
    print(dataset.answer_counter.most_common(20))
    for i in range(35, 52):
        print(dataset.games[i])



