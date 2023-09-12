# VQAClip

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

VQAClip is a machine learning model for Visual Question Answering (VQA) task.

## Install as a package
```zsh
make install
```

## Training 
### Donwload VQAv2 dataset
```zsh
make download_data
```

### Preprocess the dataset
```zsh
make preprocess
```

### Train the model
```zsh
make train
```

## Inference
### Download the trained model
```zsh
make download_model
```

### Generate text
```zsh
python3 src/inference.py --image_path [sample_image.png] --question "What is this?" --model_path model/best_model.pt
```

## Run Docker environment
### Build an image
On cpu
```zsh
docker build -t vqaclip-cpu .
```
On gpu
```zsh
docker build -t vqaclip-cuda --file Dockerfile.cuda .
```

### Run the image
```zsh
docker run -it vqaclip-cpu
```