import collections
import os
import zipfile

from generic.tf_utils.abstract_listener import EvaluatorListener
from generic.utils.file_handlers import dump_json


# TODO use an external writer instead of dumping the file in the listener
class VQADumperListener(EvaluatorListener):
    def __init__(self, tokenizer, file_path, require, use_counter=False):
        super(VQADumperListener, self).__init__(require)
        self.file_path = file_path
        self.tokenizer = tokenizer

        self.results = []
        self.counter = 0
        self.use_counter = use_counter

        self.out_path = None

    def after_batch(self, result, batch, is_training):
        for answer_id, sample in zip(result, batch['raw']):
            self.results.append({'question_id': sample.id,
                                 'answer': self.tokenizer.decode_answer(answer_id)})

    def before_epoch(self, is_training):
        self.results = []

    def after_epoch(self, is_training):
        if len(self.results):

            save_dir = os.path.dirname(self.file_path)
            json_name = os.path.basename(self.file_path)
            zip_name = "results.zip"

            # compute storage folder
            if self.use_counter:
                save_dir = os.path.join(save_dir, "res_{}".format(self.counter))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

            json_path = os.path.join(save_dir, json_name)
            zip_path = os.path.join(save_dir, zip_name)

            dump_json(json_path, self.results)
            with zipfile.ZipFile(zip_path, 'w') as myzip:
                myzip.write(json_path, json_name, zipfile.ZIP_DEFLATED)

            self.out_path(json_path)

            self.counter += 1



from vqa_eval.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval
from vqa_eval.PythonHelperTools.vqaTools.vqa import VQA

class VQAEvaluator(EvaluatorListener):
    def __init__(self, tokenizer, dump_path, ann_file, ques_file, require):
        super(VQAEvaluator, self).__init__(require)
        self.dump_path = dump_path
        self.ann_file = ann_file
        self.ques_file = ques_file
        self.tokenizer = tokenizer

        self.results = []
        self.accuracy = None


    def after_batch(self, result, batch, is_training):
        for answer_id, sample in zip(result, batch['raw']):
            self.results.append({'question_id': sample.id,
                                 'answer': self.tokenizer.decode_answer(answer_id)})

    def after_epoch(self, is_training):
        if len(self.results):
            dump_json(self.dump_path, self.results)

            vqa = VQA(self.ann_file, self.ques_file)
            vqaRes = vqa.loadRes(self.dump_path, self.ques_file)
            # create vqaEval object by taking vqa and vqaRes
            vqaEval = VQAEval(vqa, vqaRes, n=2)
            vqaEval.evaluate()

            self.accuracy = vqaEval.accuracy
            self.results = []

    def get_accuracy(self):
        return self.accuracy


