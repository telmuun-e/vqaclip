download_data:
		bash download_data.sh

download_model:
		python3 download_model.py

install:
		pip3 install --upgrade build
		python3 -m build
		pip3 install .

preprocess:
		python3 src/vqa_parser.py
		python3 src/vqa_parser.py --split valid

train:
		python3 src/train.py