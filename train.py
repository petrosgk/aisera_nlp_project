import argparse
import train_lib


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path_to_train_dataset', required=True, type=str,
                      help='Path to dataset used for training.')
  parser.add_argument('--path_to_test_dataset', required=True, type=str,
                      help='Path to dataset used for testing.')
  parser.add_argument('--outputs_dir', required=True, type=str,
                      help='Path to output directory.')
  parser.add_argument('--vocabulary_size', type=int, default=10000,
                      help='Size of vocabulary for text vectorization.')
  parser.add_argument('--max_question_length', type=int, default=20,
                      help='Max length (in words) of question inputs.')
  parser.add_argument('--max_context_length', type=int, default=250,
                      help='Max length (in words) of context inputs.')
  parser.add_argument('--embedding_dim', type=int, default=128,
                      help='Size of text embeddings.')
  parser.add_argument('--encoder_dim', type=int, default=128,
                      help='Encoder state size.')
  parser.add_argument('--features_dim', type=int, default=128,
                      help='Size of final features vector.')
  parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate.')
  parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size.')
  parser.add_argument('--max_epochs', type=int, default=100,
                      help='Max number of epochs to train for.')
  parser.add_argument('--checkpoint_frequency', type=int, default=5,
                      help='Frequency (in epochs) to save model checkpoints.')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()
  train_lib.train(path_to_train_dataset=args.path_to_train_dataset,
                  path_to_test_dataset=args.path_to_test_dataset,
                  outputs_dir=args.outputs_dir,
                  vocabulary_size=args.vocabulary_size,
                  max_question_length=args.max_question_length,
                  max_context_length=args.max_context_length,
                  embedding_dim=args.embedding_dim,
                  encoder_dim=args.encoder_dim,
                  features_dim=args.features_dim,
                  learning_rate=args.learning_rate,
                  batch_size=args.batch_size,
                  max_epochs=args.max_epochs,
                  checkpoint_frequency=args.checkpoint_frequency)
