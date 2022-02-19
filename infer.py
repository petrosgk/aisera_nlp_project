import argparse
import infer_lib


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, type=str,
                      help='Path to model checkpoint.')
  parser.add_argument('--context_txt', required=True, type=str,
                      help='Path to .txt file containing the context paragraph.')
  parser.add_argument('--question', required=True, type=str,
                      help='A question (string) the answer of which is in the context paragraph.')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()
  infer_lib.infer(checkpoint=args.checkpoint,
                  context_txt=args.context_txt,
                  question=args.question)