import pickle
import argparse
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    res = {}
    for line in tqdm(open(args.input, encoding="latin-1")):
        word, *vec = line.split()
        try:
            vec = np.asarray(list(map(float, vec)))
            res[word] = vec
        except:
            print("bad word")

    with open(args.output, 'wb') as f:
        pickle.dump(res, f)


if __name__ == '__main__':
    main()
