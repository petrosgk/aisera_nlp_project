import os
import json
import nltk
import tensorflow as tf
import numpy as np
from tqdm import tqdm


class SQUAD:
  def __init__(self,
               path_to_train_json,
               path_to_dev_json,
               vocabulary_size,
               max_question_length,
               max_context_length):
    assert os.path.basename(path_to_train_json) == 'train-v1.1.json'
    assert os.path.basename(path_to_dev_json) == 'dev-v1.1.json'
    self.path_to_train_json = path_to_train_json
    self.path_to_dev_json = path_to_dev_json
    self.max_question_length = max_question_length
    self.max_context_length = max_context_length
    self.text_vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=vocabulary_size,
                                                                      standardize=None,
                                                                      name='vectorization')
    nltk.download('punkt')

  @staticmethod
  def find_answer_start_end_in_context(context, context_words, answer_start, answer_end):
    context_token_spans = nltk.tokenize.util.align_tokens(context_words, context)
    first_idx = None
    last_idx = None
    for token_idx, token_span in enumerate(context_token_spans):
      token_start, token_end = token_span
      if token_start <= answer_start <= token_end:
        first_idx = token_idx
      if token_start <= answer_end <= token_end:
        last_idx = token_idx
    if (first_idx is None) or (last_idx is None):
      raise ValueError
    return first_idx, last_idx

  @staticmethod
  def create_text_only_dataset(dataset):
    text_only_dataset = []
    for set_entry in dataset:
      paragraphs = set_entry['paragraphs']
      for paragraph in paragraphs:
        context = paragraph['context']
        context = process_string(context)
        text_only_dataset.append(context)
        qas = paragraph['qas']
        for qas_entry in qas:
          question = qas_entry['question']
          question = process_string(question)
          text_only_dataset.append(question)
    return text_only_dataset

  def build_vocabulary_from_dataset(self, dataset):
    text_only_dataset = self.create_text_only_dataset(dataset)
    self.text_vectorization_layer.adapt(text_only_dataset)

  def save_vocabulary(self, filename):
    vocabulary = self.text_vectorization_layer.get_vocabulary(include_special_tokens=False)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
      for word in vocabulary:
        f.write(word + '\n')

  def create_tuples_dataset(self, dataset):
    tuples_dataset = []
    for set_entry in tqdm(dataset):
      paragraphs = set_entry['paragraphs']
      for paragraph in paragraphs:
        context = paragraph['context']
        context_words = nltk.word_tokenize(context)  # Split context in word tokens
        if len(context_words) > self.max_context_length:
          continue
        qas = paragraph['qas']
        for qas_entry in qas:
          question = qas_entry['question']
          question_words = nltk.word_tokenize(question)  # Split question in word tokens
          if len(question_words) > self.max_question_length:
            continue
          # Take only the first answer
          answer = qas_entry['answers'][0]
          # Start of answer as the index of a character token in the context
          answer_start = answer['answer_start']
          # Find answer end character token
          answer_end = answer_start + len(answer['text']) - 1
          # From the character level indices, we find the context word token indices that correspond to the answer
          try:
            first_idx, last_idx = self.find_answer_start_end_in_context(context, context_words, answer_start, answer_end)
          except ValueError:
            continue
          inputs = (
            self.text_vectorization_layer(process_string(context)),
            self.text_vectorization_layer(process_string(question))
          )
          outputs = (
            np.asarray([first_idx], dtype='int64'),
            np.asarray([last_idx], dtype='int64')
          )
          tuples_dataset.append((inputs, outputs))
    return tuples_dataset

  def __call__(self):
    # Load train dataset
    with open(self.path_to_train_json, 'r') as f:
      train_dataset = json.load(f)
    train_data = train_dataset['data']
    # Load dev dataset
    with open(self.path_to_dev_json, 'r') as f:
      dev_dataset = json.load(f)
    dev_data = dev_dataset['data']
    # Use training set for building the vocabulary for text vectorization
    print('Building vocabulary...')
    self.build_vocabulary_from_dataset(train_data)
    # Create datasets returning (inputs, outputs) tuples ready to be used with models
    print('Creating training set...')
    train_dataset = self.create_tuples_dataset(train_data)
    print('Creating test set...')
    test_dataset = self.create_tuples_dataset(dev_data)
    return train_dataset, test_dataset


def process_string(s):
  return ' '.join(nltk.word_tokenize(s)).lower()
