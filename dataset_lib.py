import os
import json
import string
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


class SQUAD:
  def __init__(self,
               path_to_json,
               vocabulary_size,
               max_question_length,
               max_context_length,
               test_size,
               random_state):
    assert os.path.basename(path_to_json) == 'dev-v1.1.json'
    self.path_to_json = path_to_json
    self.max_question_length = max_question_length
    self.max_context_length = max_context_length
    self.test_size = test_size
    self.random_state = random_state
    self.text_vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=vocabulary_size, name='vectorization')

  @staticmethod
  def find_answer_start_end_in_context(answer_text, context):
    answer_text_words = answer_text.split(' ')
    context_words = context.split(' ')
    try:
      first_idx = context_words.index(answer_text_words[0])
      last_idx = context_words.index(answer_text_words[-1])
    except ValueError:
      return False, False
    if (last_idx - first_idx) != (len(answer_text_words) - 1):
      return False, False
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

  def create_tuples_dataset(self, dataset):
    tuples_dataset = []
    for set_entry in dataset:
      paragraphs = set_entry['paragraphs']
      for paragraph in paragraphs:
        context = paragraph['context']
        context = process_string(context)
        num_context_words = len(context.split(' '))
        if num_context_words > self.max_context_length:
          continue
        qas = paragraph['qas']
        for qas_entry in qas:
          question = qas_entry['question']
          question = process_string(question)
          num_question_words = len(question.split(' '))
          if num_question_words > self.max_question_length:
            continue
          answers = qas_entry['answers']
          for answer in answers:
            text = answer['text']
            text = process_string(text)
            first_idx, last_idx = self.find_answer_start_end_in_context(text, context)
            if (not first_idx) or (not last_idx):
              continue
            inputs = (
              self.text_vectorization_layer(context),
              self.text_vectorization_layer(question)
            )
            outputs = (
              np.asarray([first_idx]),
              np.asarray([last_idx])
            )
            tuples_dataset.append((inputs, outputs))
    return tuples_dataset

  def __call__(self):
    with open(self.path_to_json, 'r') as f:
      dataset = json.load(f)
    data = dataset['data']
    train_set, test_set = train_test_split(data, test_size=self.test_size, random_state=self.random_state)
    # Use training set for building the vocabulary for text vectorization
    print('Building vocabulary...')
    self.build_vocabulary_from_dataset(train_set)
    # Create datasets returning (inputs, outputs) tuples ready to be used with models
    print('Creating training set...')
    train_dataset = self.create_tuples_dataset(train_set)
    print('Creating test set...')
    test_dataset = self.create_tuples_dataset(test_set)
    return train_dataset, test_dataset


def process_string(s):
  processed_s = s.lower()
  for c in string.punctuation:
    processed_s = processed_s.replace(c, ' ')
  while '  ' in processed_s:
    processed_s = processed_s.replace('  ', ' ')
  processed_s = processed_s.strip()
  return processed_s