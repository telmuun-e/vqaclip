import argparse
import pickle
import json
from tqdm import tqdm
from PIL import Image
import torch
from PIL import Image
import open_clip


class VQAParser:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', device=args.device)
        question_filename = "v2_OpenEnded_mscoco_train2014_questions.json" if args.split == "train" else "v2_OpenEnded_mscoco_val2014_questions.json"
        answer_filename = "v2_mscoco_train2014_annotations.json" if args.split == "train" else "v2_mscoco_val2014_annotations.json"
        self.images_filepath = "train2014/COCO_train2014_" if args.split == "train" else "val2014/COCO_val2014_"

        with open(f"{args.input_path}/{args.split}/{question_filename}", "r") as f:
            self.questions = json.load(f)["questions"]
        with open(f"{args.input_path}/{args.split}/{answer_filename}", "r") as f:
            self.answers = json.load(f)["annotations"]


    def parse(self) -> None:
        answers_by_question = self.parse_answers()
        answers = []
        questions = []
        clip_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(len(self.questions))):
                question = self.questions[i]
                question_id = question["question_id"]
                questions.append(question["question"])
                answers.append(answers_by_question[question_id])

                image_id = question["image_id"]
                image_path = f"{self.args.input_path}/{self.args.split}/{self.images_filepath}{int(image_id):012d}.jpg"
                image = self.preprocess(Image.open(image_path)).unsqueeze(0)
                image_features = self.model.encode_image(image.to(self.args.device))
                clip_embeddings.append(image_features[0].cpu())
                
        with open(f"{self.args.output_path}/{self.args.split}/data.pkl", "wb") as f:
            pickle.dump({"questions": questions, "answers": answers, "clip_embeddings": clip_embeddings}, f)


    def parse_answers(self):
        answers_by_question = {}
        for answers in self.answers:
            answers_by_question[answers["question_id"]] = [entry["answer"] for entry in answers["answers"]]
        return answers_by_question



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default="data/raw")
    parser.add_argument('--output_path', default="data/preprocessed")
    parser.add_argument('--split', default="train")
    parser.add_argument('--device', default="cuda")
    args = parser.parse_args()
    vqa_parser = VQAParser(args)
    vqa_parser.parse()
