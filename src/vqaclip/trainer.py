import os
import datetime
from time import time, ctime
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer

from vqaclip.model.vqaclip import VQAClip
from vqaclip.dataset import VQADataset
from vqaclip.logger import IOStream
from vqaclip.utils import generate


class Trainer:
    def __init__(
            self, input_path: str, output_path: str, max_len: int, 
            prefix_length: int, prefix_size: int, clip_length: int, 
            num_layers: int, batch_size: int, n_epochs: int, 
            learning_rate: int, device: str
            ) -> None:
        self.n_epochs = n_epochs
        self.device = device
        self.prefix_length = prefix_length
        self.output_path = output_path
        self.min_loss = float("inf")

        self.io = IOStream(output_path + "/run_" + datetime.datetime.now().strftime("%Y%m%d%h%m") + ".log")
        self.io.cprint('Program start: %s' % ctime(time()))
        
        train_dataset = VQADataset(f"{input_path}/train/data.pkl", max_len, mode="train")
        self.train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)

        valid_dataset = VQADataset(f"{input_path}/valid/data.pkl", max_len, mode="train")
        self.valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=2)

        test_dataset = VQADataset(f"{input_path}/valid/data.pkl", max_len, mode="eval")
        self.test_dataloader = DataLoader(test_dataset, 1, shuffle=False, num_workers=1)

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = VQAClip(max_len, prefix_length, prefix_size, clip_length, num_layers)
        self.model.to(device)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=1000, num_training_steps=n_epochs * len(self.train_dataloader)
        )

    
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0
        for batch in self.valid_dataloader:
            with torch.no_grad():
                outputs = self.model(batch["prefix"].to(self.device), batch["input_tokens"].to(self.device))
            target_logits = [
                    logits[l1: l2] for logits, l1, l2 in zip(
                        outputs.logits,
                        self.prefix_length + batch["question_len"] - 1,
                        self.prefix_length + batch["question_len"] + batch["answer_len"]
                    )
                ]

            target_tokens = [
                tokens[l1: l2] for tokens, l1, l2 in zip(
                    batch["input_tokens"].to(self.device),
                    batch["question_len"],
                    batch["question_len"] + batch["answer_len"] + 1
                )
            ]

            loss = F.cross_entropy(torch.cat(target_logits), torch.cat(target_tokens))
            total_loss += loss.item()

        self.io.cprint(f"Validation loss: {total_loss}")
        self.compute_accuracy()
        return total_loss

    def compute_accuracy(self) -> None:
        labels = []
        for i, batch in enumerate(self.test_dataloader):
            generated_text = generate(self.model, self.tokenizer, batch["questions"][0], batch["prefix"].to(self.device), self.prefix_length, device=self.device)
            target = batch["answers"]
            target = np.array(target).T.tolist()[0]      
            labels.append(int(generated_text in target))
            
        acc_score = sum(labels) / len(labels)
        self.io.cprint(f"Accuracy: {acc_score}")


    def train(self) -> None :
        self.model.train()
        for epoch in range(self.n_epochs):
            self.io.cprint(f"Epoch: {epoch+1}")
            total_loss = 0
            self.model.zero_grad()
            for i, batch in enumerate(self.train_dataloader):
                outputs = self.model(batch["prefix"].to(self.device), batch["input_tokens"].to(self.device))

                target_logits = [
                    logits[l1: l2] for logits, l1, l2 in zip(
                        outputs.logits,
                        self.prefix_length + batch["question_len"] - 1,
                        self.prefix_length + batch["question_len"] + batch["answer_len"]
                    )
                ]

                target_tokens = [
                    tokens[l1: l2] for tokens, l1, l2 in zip(
                        batch["input_tokens"].to(self.device),
                        batch["question_len"],
                        batch["question_len"] + batch["answer_len"] + 1
                    )
                ]

                loss = F.cross_entropy(torch.cat(target_logits), torch.cat(target_tokens))
                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if (i+1) % 10000 == 0:
                    self.io.cprint(f"Training total loss: {total_loss}")
                    total_loss = 0
            loss = self.evaluate()
            self.save_model_state(loss)


    def save_model_state(self, loss: float) -> None:
        self.io.cprint("Saving new checkpoint")
        checkpoint_dir = self.output_path + "/checkpoint"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        filename = "checkpoint_" + datetime.datetime.now().strftime("%Y%m%d%h%M") + ".pt"
        torch.save(self.model.state_dict(), checkpoint_dir + "/" + filename)
        if loss <= self.min_loss:
            self.io.cprint(f"New low loss: {loss}")
            self.min_loss = loss
            torch.save(self.model.state_dict(), self.output_path + "/best_model.pt")

                
    def load_model_state(self) -> None:
        self.io.cprint("Loading model weight")
        self.model.load_state_dict(torch.load(self.output_path + "/best_model.pt"))