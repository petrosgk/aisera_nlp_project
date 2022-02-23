import numpy as np
import tensorflow as tf
import nltk
import dataset_lib


def infer(checkpoint, context_txt, question):
  # Load model
  model = tf.keras.models.load_model(checkpoint)
  # Load text file containing context paragraph
  with open(context_txt, 'r', encoding='utf-8') as f:
    context = f.read()
  print(f'Context: "{context}"')
  print(f'Question: "{question}"')
  # Process context and question
  processed_context = dataset_lib.process_string(context)
  processed_question = dataset_lib.process_string(question)
  print(f'Processed context: "{processed_context}"')
  print(f'Processed question: "{processed_question}"')
  # Inputs to the model are the context P and question Q
  # Outputs are the probabilities of the indexes of the start (a_s) and end (a_e) tokens of the answer in P
  inputs = (np.asarray([processed_context]), np.asarray([processed_question]))
  start_idx_probs, end_idx_probs = model.predict_on_batch(inputs)
  start_idx = np.argmax(start_idx_probs)
  end_idx = np.argmax(end_idx_probs)
  # Find the first and last word of the answer in the context
  context_words = nltk.word_tokenize(context)
  assert len(context_words) > end_idx >= start_idx and start_idx < len(context_words)
  answer_words = context_words[start_idx:end_idx + 1]
  answer = ' '.join(answer_words)
  print(f'Answer: "{answer}"')