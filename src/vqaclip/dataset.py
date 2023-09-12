import pickle
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class VQADataset(Dataset):
    def __init__(self, data_path: str, max_len: int, mode: str = "train") -> None:
        self.max_len = max_len
        self.mode = mode
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        self.questions_tokens = self.tokenizer(data["questions"])["input_ids"]
        self.clip_embeddings = data["clip_embeddings"]
        self.answers = data["answers"]
        self.questions = data["questions"]


    def __len__(self) -> int:
        return len(self.questions_tokens)


    def __getitem__(self, item: int) -> dict:
        prefix = self.clip_embeddings[item].type(torch.float)
        question_tokens = torch.tensor(self.questions_tokens[item])
        question_len = question_tokens.shape[0]
        eos_token = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.int64)

        if self.mode == "predict":
            return {
                "prefix": prefix,
                "input_tokens": question_tokens,
                "question_len": question_len,
            }
        
        if self.mode == "eval":
            answers = self.answers[item]
            answers_len = len(answers)
            dummy = ["NaN" for _ in range(15 - len(answers))]
            answers.extend(dummy)
            padding = self.max_len - (question_tokens.shape[0])
            padding = torch.zeros(padding, dtype=torch.int64)
            
            input_tokens = torch.cat([question_tokens, eos_token, padding])
            return {
                "prefix": prefix,
                "input_tokens": input_tokens,
                "question_len": question_len,
                "answers": answers,
                "answers_len": answers_len,
                "questions": self.questions[item]
            }

        answer = random.sample(self.answers[item], 1)[0]
        answer_tokens = torch.tensor(self.tokenizer(answer)["input_ids"], dtype=torch.int64)
        answer_len = answer_tokens.shape[0]
        
        padding = self.max_len - (question_tokens.shape[0] + answer_tokens.shape[0])
        padding = torch.zeros(padding, dtype=torch.int64)
        input_tokens = torch.cat([question_tokens, answer_tokens, eos_token, padding])

        return {
            "prefix" : prefix,
            "input_tokens" : input_tokens,
            "question_len": question_len,
            "answer_len": answer_len
        }
        