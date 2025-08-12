import os
import sys
import argparse

# Import the modules
import subprocess


def run_preprocess():
    print('[main.py] Running data preprocessing...')
    # Call the preprocess script
    ret = subprocess.call([sys.executable, os.path.join('src', 'preprocess.py')])
    if ret != 0:
        print('[main.py] Preprocessing failed.')
        sys.exit(1)


def run_train():
    print('[main.py] Running training...')
    ret = subprocess.call([sys.executable, os.path.join('src', 'train.py')])
    if ret != 0:
        print('[main.py] Training failed.')
        sys.exit(1)


def run_evaluate():
    print('[main.py] Running evaluation...')
    ret = subprocess.call([sys.executable, os.path.join('src', 'evaluate.py')])
    if ret != 0:
        print('[main.py] Evaluation failed.')
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Run full experiment pipeline: preprocess, train, evaluate.')
    parser.add_argument('--skip-preprocess', action='store_true', help='Skip preprocessing step')
    parser.add_argument('--only', choices=['preprocess', 'train', 'evaluate'], help='Run only one phase')
    args = parser.parse_args()

    if args.only:
        if args.only == 'preprocess':
            run_preprocess()
        elif args.only == 'train':
            run_train()
        elif args.only == 'evaluate':
            run_evaluate()
    else:
        if not args.skip_preprocess:
            run_preprocess()
        run_train()
        run_evaluate()
    print('[main.py] Experiment pipeline completed.')

if __name__ == '__main__':
    main()
