import argparse
from train import train_main, parse_train_args
from test import test_main, parse_test_args

def main():
    parser = argparse.ArgumentParser(description="Main entry point.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                        help="Choose whether to train or test the model.")
    args, remaining_args = parser.parse_known_args()

    if args.mode == "train":
        # Re-parse using train arguments
        train_args = parse_train_args()
        train_main(train_args)
    elif args.mode == "test":
        # Re-parse using test arguments
        test_args = parse_test_args()
        test_main(test_args)

if __name__ == "__main__":
    main()
