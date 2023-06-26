import glob
import sys
import json
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".",
        help="path where all the json files are located")

    parser.add_argument("--output_file", type=str, default="merged_output.json",
        help="filename where the merged json should go")

    args = parser.parse_args()

    data_path = args.data_path
    out_file = args.output_file

    text_files = glob.glob(data_path + '/*.txt')

    counter = 0

    with open(out_file, 'w') as outfile:
        for fname in text_files:
            counter += 1

            if counter % 1024 == 0:
                print("Merging at ", counter, flush=True)

            with open(fname, 'r') as infile:
                for row in infile:
                    tmp = {}
                    tmp['text'] = row
                    outfile.write(json.dumps(tmp))
                    outfile.write('\n')


    print("Merged file", out_file, flush=True)

