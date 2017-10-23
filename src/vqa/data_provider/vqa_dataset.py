import collections
import json

from generic.data_provider.dataset import AbstractDataset
from vqa_eval.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval
use_100 = False

class Image:
    def __init__(self, img_id, image_builder, year, which_set):
        self.id = img_id
        self.url = "http://mscoco.org/images/{}".format(img_id)

        if image_builder is not None:
            filename = "{set}{year}/COCO_{set}{year}_{id}.jpg".format(
                set=which_set, year=year, id=str(img_id).zfill(12))
            self.image_loader = image_builder.build(img_id,
                                                    filename=filename,
                                                    which_set=which_set,
                                                    optional=False)

    def get_image(self):
        return self.image_loader.get_image()


class Game(object):
    def __init__(self, game_id, image, question, majority_answer, answers, question_type, answer_type):
        self.id = game_id
        self.image = image
        self.question = question
        self.majority_answer = majority_answer
        self.answers = answers
        self.question_type = question_type
        self.answer_type = answer_type

    def __str__(self):
        return "[#q:{}, #p:{}] {} - {} ({})".format(self.id, self.image.id, self.question, self.majority_answer, self.answer_type)


# VQA have some specific answer preprocessing that we apply here
dummy_vqa = lambda: None
dummy_vqa.getQuesIds = lambda: None
vqa_eval = VQAEval(dummy_vqa, None)
def process_answers(answer):

    answer = answer.replace('\n', ' ')
    answer = answer.replace('\t', ' ')
    answer = answer.strip()
    answer = vqa_eval.processPunctuation(answer)
    answer = vqa_eval.processDigitArticle(answer)

    return answer



class VQADataset(AbstractDataset):
    """Loads the dataset."""

    def __init__(self, folder, year, which_set, image_builder=None, preprocess_answers=False):

        annotations_path_file = '{}/vqa_{}{}_annotations.json'.format(folder, which_set, year)
        questions_path_file = '{}/vqa_{}{}_questions.json'.format(folder, which_set, year)

        games = []
        self.question_types = collections.Counter()

        self.answer_counter = collections.Counter()
        self.answer_types = collections.Counter()

        # Load questions
        print("Loading questions...")
        with open(questions_path_file) as questions_file:
            full_questions = json.load(questions_file)

        # Load annotations - train/val only
        try:
            print("Loading annotations...")
            with open(annotations_path_file) as annotations_file:
                full_annotations = json.load(annotations_file)

            assert full_annotations["info"]["version"] == full_questions["info"]["version"]
            assert full_annotations["data_subtype"] == full_questions["data_subtype"]
            assert full_annotations["data_subtype"].startswith(which_set)
            assert full_annotations["data_subtype"].endswith(str(year))

        except FileNotFoundError:
            print("No annotations file... (Test dataset)")
            assert "test" in which_set

            # as test-year is always 2015, it is easier to hard-code it
            year = 2015

            # Create a dummy annotation file for test dataset
            full_annotations = []
            for q in full_questions["questions"]:
                full_annotations.append({
                    "question_id" : q["question_id"],
                    "question_type": "unk",
                    "multiple_choice_answer" : "unk",
                    "answers" : [{"answer" : "unk"}],
                    "answer_type" : "unk"
                })
            full_annotations = {"annotations" : full_annotations}

        print("Starting generating VQA v{} dataset ({})".format(full_questions["info"]["version"], which_set))

        # Start creating the dataser
        for annotation, question in zip(full_annotations["annotations"], full_questions["questions"]):
            assert annotation["question_id"] == question["question_id"]

            question_id = int(question["question_id"])
            image_id = question["image_id"]

            question = question["question"]
            question_type = annotation["question_type"]
            self.question_types[question_type] += 1

            majority_answer = annotation["multiple_choice_answer"]
            answers = [ a["answer"] for a in annotation["answers"]]
            answer_type = annotation["answer_type"]

            if preprocess_answers:
                majority_answer = process_answers(majority_answer)
                answers = [process_answers(a) for a in answers]

            for a in answers:
                self.answer_counter[a] += 1
            self.answer_types[answer_type] += 1

            games.append(Game(game_id=question_id,
                              image=Image(image_id, image_builder, year, which_set),
                              question=question,
                              question_type=question_type,
                              majority_answer=majority_answer,
                              answers=answers,
                              answer_type=answer_type))

            if use_100 and len(games) > 100: break

        print('{} games loaded...'.format(len(games)))
        super(VQADataset, self).__init__(games)



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



