import pandas as pd
import tensorflow as tf
from pathlib import Path

import config
from model import getBERTModel
from data import BertSemanticDataGenerator
import keras

# Load CSV data using panda
train_df = pd.read_csv(config.dirs["training_csv_file"])
test_df = pd.read_csv(config.dirs["validation_csv_file"])
valid_df = pd.read_csv(config.dirs["test_csv_file"])

# Check data size
print(f"Total train samples : {train_df.shape[0]}")
print(f"Total validation samples: {valid_df.shape[0]}")
print(f"Total test samples: {valid_df.shape[0]}")

train_df.dropna(axis=0, inplace=True)

train_df = (
    train_df[train_df.label != "-"]
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)

valid_df = (
    valid_df[valid_df.label != "-"]
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)

train_data = BertSemanticDataGenerator(
    train_df["text"].values.astype("str"),
    train_df.label,
    batch_size=config.dl_train_params['batch_size'],
    shuffle=config.dl_train_params['shuffle'],
    max_length=config.dl_train_params['max_length'],
    num_classes=config.dl_train_params['n_classes'],
    transformers_model_name = config.training_params['transformers_model_name']
)

valid_data = BertSemanticDataGenerator(
    valid_df["text"].values.astype("str"),
    valid_df.label,
    batch_size=config.dl_eval_params['batch_size'],
    shuffle=config.dl_eval_params['shuffle'],
    max_length=config.dl_eval_params['max_length'],
    num_classes=config.dl_eval_params['n_classes'],
    transformers_model_name = config.eval_params['transformers_model_name']
)

# Create model
model = getBERTModel(max_length=config.dl_eval_params['max_length'], 
    num_classes=config.dl_train_params['n_classes'],
    transformers_model_name = config.training_params['transformers_model_name']
)
print(model.summary())

# Step 1: Training with BERT variables freezed
model.compile(
    optimizer=config.training_params['optimizer'],
    loss=config.training_params['loss'],
    metrics=config.training_params['metrics']
)
model.fit(
    train_data,
    validation_data=valid_data,
    epochs=config.training_params['epochs'],
    use_multiprocessing=config.training_params['multiprocessing'],
    workers=-1,
)

# Step 2: Training with BERT variables unfreezed
model.get_layer('bert').trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=config.training_params['loss'],
    metrics=config.training_params['metrics']
)
model.fit(
    train_data,
    validation_data=valid_data,
    epochs=config.training_params['epochs'],
    use_multiprocessing=config.training_params['multiprocessing'],
    workers=-1,
)

# Save model in protobuff format
print("Saving BERT model to "+ config.dirs['results_path']+"/model")
Path(config.dirs['results_path']).mkdir(parents=True, exist_ok=True)
model.save(config.dirs['results_path']+"/model")










