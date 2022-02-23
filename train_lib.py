import os
import math
import numpy as np
import tensorflow as tf
import dataset_lib
import model_lib


class DataSequence(tf.keras.utils.Sequence):
  def __init__(self, dataset, batch_size):
    self.dataset = dataset
    self.batch_size = batch_size

  @staticmethod
  def create_batches(data):
    context_batch = []
    question_batch = []
    first_idx_batch = []
    last_idx_batch = []
    for entry in data:
      inputs, outputs = entry
      context, question = inputs
      context_batch.append(context)
      question_batch.append(question)
      first_idx, last_idx = outputs
      first_idx_batch.append(first_idx)
      last_idx_batch.append(last_idx)
    context_batch = tf.keras.preprocessing.sequence.pad_sequences(sequences=context_batch, padding='post', dtype='int64')
    question_batch = tf.keras.preprocessing.sequence.pad_sequences(sequences=question_batch, padding='post', dtype='int64')
    first_idx_batch = np.concatenate(first_idx_batch, axis=0)
    last_idx_batch = np.concatenate(last_idx_batch, axis=0)
    inputs = (context_batch, question_batch)
    outputs = (first_idx_batch, last_idx_batch)
    return inputs, outputs

  def __len__(self):
    return math.ceil(len(self.dataset) / self.batch_size)

  def __getitem__(self, idx):
    data = self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]
    inputs, outputs = self.create_batches(data)
    return inputs, outputs


class ModelCheckpoint(tf.keras.callbacks.Callback):
  def __init__(self,
               checkpoints_dir,
               model,
               frequency,):
    self.checkpoints_dir = checkpoints_dir
    self.frequency = frequency
    print('Creating inference model...')
    self.inference_model = model.create_inference_model()

  def on_epoch_end(self, epoch, logs=None):
    if (epoch + 1) % self.frequency == 0:
      for layer in self.model.layers:
        self.inference_model.get_layer(layer.name).set_weights(layer.get_weights())
      loss = round(logs['loss'], ndigits=4)
      val_loss = round(logs['val_loss'], ndigits=4)
      checkpoint_path = os.path.join(self.checkpoints_dir, f'epoch_{epoch}.loss_{loss}.val_loss_{val_loss}')
      print('\nSaving inference model...')
      self.inference_model.save(checkpoint_path)


def train(path_to_train_dataset,
          path_to_test_dataset,
          outputs_dir,
          max_question_length,
          max_context_length,
          vocabulary_size,
          embedding_dim,
          encoder_dim,
          features_dim,
          learning_rate,
          batch_size,
          max_epochs,
          checkpoint_frequency):
  # Initialize SQUAD dataset processor
  print('Creating train and test sets...')
  squad_dataset = dataset_lib.SQUAD(path_to_train_json=path_to_train_dataset,
                                    path_to_dev_json=path_to_test_dataset,
                                    vocabulary_size=vocabulary_size,
                                    max_question_length=max_question_length,
                                    max_context_length=max_context_length)
  # Create train and test sets
  train_set, test_set = squad_dataset()
  # Save vocabulary to file
  squad_dataset.save_vocabulary(os.path.join(outputs_dir, 'squad.vocabulary.txt'))
  # Create model used for training/validation
  print('Creating model...')
  model = model_lib.Model(vocabulary_size=vocabulary_size,
                          max_context_length=max_context_length,
                          embedding_dim=embedding_dim,
                          encoder_dim=encoder_dim,
                          features_dim=features_dim,
                          vectorization_layer=squad_dataset.text_vectorization_layer)
  train_model = model.create_model()
  train_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
  # Create the data sequences that will feed the model
  train_data_sequence = DataSequence(dataset=train_set, batch_size=batch_size)
  test_data_sequence = DataSequence(dataset=test_set, batch_size=batch_size)
  # Define callbacks
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(outputs_dir, 'logs')),
    ModelCheckpoint(checkpoints_dir=os.path.join(outputs_dir, 'checkpoints'),
                    model=model,
                    frequency=checkpoint_frequency)
  ]
  # Train
  print('Training...')
  train_model.fit(x=train_data_sequence,
                  validation_data=test_data_sequence,
                  epochs=max_epochs,
                  callbacks=callbacks,
                  verbose=1)
