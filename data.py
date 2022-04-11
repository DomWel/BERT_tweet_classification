import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import keras
transformers.logging.set_verbosity_error()

# Source code is largely adapted from: 
# https://keras.io/examples/nlp/semantic_similarity_with_bert/
class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=32,
        shuffle=True,
        include_targets=True,
        max_length = 128,
        num_classes  = 3,
        transformers_model_name = "bert-base-german-cased"
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.max_length = max_length
        self.include_targets = include_targets
        self.num_classes = num_classes
        self.transformers_model_name = transformers_model_name
        # Load BERT Tokenizer to encode the text.
        # Has to be compatible with the pretrained BERT model in use!!!!!
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            self.transformers_model_name, do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        if self.include_targets:
            label = tf.keras.utils.to_categorical(self.labels[indexes], num_classes=self.num_classes)
            labels = np.array(label, dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)
