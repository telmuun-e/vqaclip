# Train images
TRAIN_IMAGES_PATH=data/raw/train/train_image.zip
curl http://images.cocodataset.org/zips/train2014.zip --output $TRAIN_IMAGES_PATH
unzip $TRAIN_IMAGES_PATH -d data/raw/train
rm $TRAIN_IMAGES_PATH

# Train questions
TRAIN_QUESTIONS_PATH=data/raw/train/train_questions.zip
curl https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip --output $TRAIN_QUESTIONS_PATH
unzip $TRAIN_QUESTIONS_PATH -d data/raw/train
rm $TRAIN_QUESTIONS_PATH

# Train answers
TRAIN_ANSWERS_PATH=data/raw/train/train_annotations.zip
curl https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip --output $TRAIN_ANSWERS_PATH
unzip $TRAIN_ANSWERS_PATH -d data/raw/train
rm $TRAIN_ANSWERS_PATH

# Valid images
VALID_IMAGES_PATH=data/raw/train/valid_image.zip
curl http://images.cocodataset.org/zips/val2014.zip --output $VALID_IMAGES_PATH
unzip $VALID_IMAGES_PATH -d data/raw/valid
rm $VALID_IMAGES_PATH

# Valid questions
VALID_QUESTIONS_PATH=data/raw/valid/valid_questions.zip
curl https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip --output $VALID_QUESTIONS_PATH
unzip $VALID_QUESTIONS_PATH -d data/raw/valid
rm $VALID_QUESTIONS_PATH

# Valid answers
VALID_ANSWERS_PATH=data/raw/train/train_annotations.zip
curl https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip --output $VALID_ANSWERS_PATH
unzip $VALID_ANSWERS_PATH -d data/raw/valid
rm $VALID_ANSWERS_PATH
