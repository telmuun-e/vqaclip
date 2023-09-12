import argparse
from vqaclip.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default="data/preprocessed")
    parser.add_argument('--output_path', default="model")
    parser.add_argument('--max_len', type=int, default=80)
    parser.add_argument('--prefix_length', type=int, default=40)
    parser.add_argument('--prefix_size', type=int, default=1024)
    parser.add_argument('--clip_length', type=int, default=40)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--device', default="cuda")
    args = parser.parse_args()

    trainer = Trainer(
        args.input_path, 
        args.output_path, 
        args.max_len, 
        args.prefix_length, 
        args.prefix_size, 
        args.clip_length, 
        args.num_layers, 
        args.batch_size, 
        args.n_epochs,
        args.learning_rate,
        args.device
        )
    
    trainer.train()


if __name__ == "__main__":
    main()