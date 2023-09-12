import argparse
from PIL import Image
import torch
from transformers import GPT2Tokenizer
import open_clip

from vqaclip.model.vqaclip import VQAClip
from vqaclip.utils import generate


class Inference:
    def __init__(self, cfg: dict) -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_length = cfg["clip_length"]
        self.clip_encoder, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', device=self.device)
        self.model = VQAClip(cfg["max_len"], cfg["prefix_length"], cfg["prefix_size"], cfg["clip_length"], cfg["num_layers"])
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(cfg["model_path"]))
        self.model.eval()

        self.max_len = cfg["max_len"]
        self.prefix_length = cfg["prefix_length"]


    def generate_text(self, image: Image.Image, question: str) -> str:
        image = self.clip_preprocess(image).unsqueeze(0)
        image_features = self.clip_encoder.encode_image(image.to(self.device))
        generated_text = generate(self.model, self.tokenizer, question, image_features, self.clip_length, device=self.device)
        return generated_text
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path')
    parser.add_argument('--question')
    parser.add_argument('--model_path')
    args = parser.parse_args()

    image = Image.open(args.image_path)
    question = args.question

    cfg = {
        "max_len": 80,
        "prefix_length": 40,
        "prefix_size": 1024,
        "clip_length": 40,
        "num_layers": 8,
        "model_path": args.model_path,
    }

    inference = Inference(cfg)
    print(inference.generate_text(image, question))