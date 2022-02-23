import tensorflow as tf


class Model:
  def __init__(self,
               vocabulary_size,
               embedding_dim,
               encoder_dim,
               features_dim,
               max_context_length,
               vectorization_layer=None):
    self.vocabulary_size = vocabulary_size
    self.embedding_dim = embedding_dim
    self.encoder_dim = encoder_dim
    self.features_dim = features_dim
    self.max_context_length = max_context_length
    self.vectorization_layer = vectorization_layer

    # Shared embedding layer
    self.embedding = tf.keras.layers.Embedding(input_dim=self.vocabulary_size + 1,
                                               output_dim=self.embedding_dim,
                                               mask_zero=True,
                                               name='embedding')
    # Shared encoder layers
    self.rnn_0 = tf.keras.layers.Bidirectional(
      tf.keras.layers.LSTM(units=self.encoder_dim, return_sequences=True, return_state=True),
      name='encoder_rnn_0'
    )
    self.rnn_1 = tf.keras.layers.Bidirectional(
      tf.keras.layers.LSTM(units=self.encoder_dim, return_sequences=True),
      name='encoder_rnn_1'
    )
    self.pool = tf.keras.layers.GlobalAveragePooling1D(name='pool')

  def extract_features(self, inputs):
    x = self.embedding(inputs)
    # shape: (batch_size, sequence_length, embedding_dim)
    x = self.rnn_0(x)
    # shape: (batch_size, sequence_length, 2 * encoder_dim)
    x = self.rnn_1(x)
    # shape: (batch_size, sequence_length, 2 * encoder_dim)
    x = self.pool(x)
    # shape: (batch_size, 2 * encoder_dim)
    return x

  def merge_features(self, context_features, question_features):
    x = tf.keras.layers.Concatenate(axis=-1, name='concat')([context_features, question_features])
    # shape: (batch_size, 4 * encoder_dim)
    x = tf.keras.layers.Dense(units=self.features_dim, activation='tanh', name='features')(x)
    # shape: (batch_size, features_dim)
    return x

  def process(self, context, question):
    context_features = self.extract_features(context)
    # shape: (batch_size, 2 * encoder_dim)
    question_features = self.extract_features(question)
    # shape: (batch_size, 2 * encoder_dim)
    features = self.merge_features(context_features, question_features)
    # shape: (batch_size, features_dim)
    start_index = tf.keras.layers.Dense(units=self.max_context_length, activation='softmax', name='start_index')(features)
    # shape: (batch_size, max_context_length)
    end_index = tf.keras.layers.Dense(units=self.max_context_length, activation='softmax', name='end_index')(features)
    # shape: (batch_size, max_context_length)
    return start_index, end_index

  def create_model(self):
    context = tf.keras.layers.Input(shape=(None,), dtype='int64', name='context_inputs')
    # shape: (batch_size, context_sequence_length)
    question = tf.keras.layers.Input(shape=(None,), dtype='int64', name='question_inputs')
    # shape: (batch_size, question_sequence_length)
    start_index, end_index = self.process(context, question)
    # shape: [(batch_size, max_context_length), (batch_size, max_context_length)]
    model = tf.keras.Model(inputs=[context, question], outputs=[start_index, end_index])
    print(model.summary())
    return model

  def create_inference_model(self):
    context = tf.keras.layers.Input(shape=(), dtype='string', name='context_inputs')
    # shape: (batch_size, context_sequence_length)
    question = tf.keras.layers.Input(shape=(), dtype='string', name='question_inputs')
    # shape: (batch_size, question_sequence_length)
    vectorized_context = self.vectorization_layer(context)
    # shape: (batch_size, context_sequence_length)
    vectorized_question = self.vectorization_layer(question)
    # shape: (batch_size, question_sequence_length)
    start_index, end_index = self.process(vectorized_context, vectorized_question)
    # shape: [(batch_size, max_context_length), (batch_size, max_context_length)]
    model = tf.keras.Model(inputs=[context, question], outputs=[start_index, end_index])
    print(model.summary())
    return model
