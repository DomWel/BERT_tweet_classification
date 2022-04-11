import pandas as pd
import keras
from sklearn.metrics import classification_report
import numpy as np
from data import BertSemanticDataGenerator
import config

# Load model
model = keras.models.load_model(config.dirs["results_path"]+"/model")

# Load CSV test data file
test_df = pd.read_csv(config.dirs["test_csv_file"])
print(test_df['label'])

test_data = BertSemanticDataGenerator(
    test_df["text"].values.astype("str"),
    test_df.label,
    batch_size=config.dl_eval_params['batch_size'],
    shuffle=config.dl_eval_params['shuffle'],
    max_length=config.dl_eval_params['max_length'],
    num_classes=config.dl_eval_params['n_classes'], 
    transformers_model_name = config.eval_params['transformers_model_name']
)

model.compile(
    loss=config.eval_params['loss'],
    metrics=config.eval_params['metrics']
)

#results = model.evaluate(test_data, verbose=1)

# Get per-class precision
y_pred = model.predict(test_data, verbose=1)
y_pred = np.argmax(y_pred, axis=1) # Convert one-hot to index

print(classification_report(y_pred, test_df['label']))




