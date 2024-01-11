import os
from tqdm import tqdm
import tensorflow as tf

try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass
from keras.models import load_model

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.tensorflow.dataProvider import DataProvider
from mltu.preprocessors import ImageReader
from mltu.annotations.images import CVImage
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric

import re

from models import text_recognition
from configs import ModelConfigs

configs = ModelConfigs()

def sorted_alphanumeric(data):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def create_dataset(dataset_path):
    words = open(os.path.join(dataset_path, "labels.txt"), "r").readlines()
    dataset = []
    vocab =  set()
    max_len = 0

    for index, filename in enumerate(sorted_alphanumeric(os.listdir(dataset_path))):
        if filename.endswith('.jpg'):
            label = words[index].replace('\n', '')
            dataset.append([os.path.join(dataset_path, filename), label])
            vocab.update(list(label))
            max_len = max(max_len, len(label))
            
    vocab = sorted(vocab)
    return dataset, vocab, max_len

train_dataset, train_vocab, max_train_len = create_dataset('./Datasets/train')
val_dataset, val_vocab, max_val_len       = create_dataset('./Datasets/validation')


configs.vocab = "".join(train_vocab)
configs.max_text_length = max(max_train_len, max_val_len)
configs.save()


print(train_vocab)

# Create training data provider
train_data_provider = DataProvider(
    dataset=train_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
)

# Create validation data provider
val_data_provider = DataProvider(
    dataset=val_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
)

model = text_recognition(
    input_dim = (configs.height, configs.width, 1),
    output_dim = len(configs.vocab),
)
# # Compile the model and print summary
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate), 
    loss=CTCloss(), 
    metrics=[CWERMetric(padding_token=len(configs.vocab))],
    run_eagerly=False
)

# uncomment it if you want to retrain the model
# model = load_model('./Models/phone_reader.h5', custom_objects={'CTCloss': CTCloss, 'CWERMetric': CWERMetric})


os.makedirs(configs.model_path, exist_ok=True)


earlystopper = EarlyStopping(monitor="val_CER", patience=50, verbose=1)
checkpoint = ModelCheckpoint(f"{configs.model_path}/phone_reader.h5", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=5, verbose=1, mode="auto")

model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback],
    workers=configs.train_workers,
)


