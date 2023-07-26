import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # default arguments
    parser.add_argument('--date', help='date of video (YYYY-MM-DD)', type=str, default='2023-07-26')
    args = parser.parse_args()
    return args