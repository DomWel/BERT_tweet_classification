import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import keras

from data import BertSemanticDataGenerator
import config

# Load model
model = keras.models.load_model(config.dirs["results_path"]+"/model")

# Load CSV test data file
test_df = pd.read_csv(config.dirs["test_csv_file"], nrows=100000)

test_data = BertSemanticDataGenerator(
    test_df["text"].values.astype("str"),
    test_df.label,
    batch_size=config.dl_eval_params['batch_size'],
    shuffle=config.dl_eval_params['shuffle'],
    max_length=config.dl_eval_params['max_length'],
    num_classes=config.dl_eval_params['n_classes']
)

model.compile(
    loss=config.eval_params['loss'],
    metrics=config.eval_params['metrics']
)

results = model.evaluate(test_data, verbose=1)






