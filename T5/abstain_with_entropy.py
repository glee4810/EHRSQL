import os
import json
import argparse
import warnings

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--infernece_result_path', required=True, type=str, help='path for inference')
    args.add_argument('--input_file', default='prediction_raw.json', type=str, help='path for inference')
    args.add_argument('--output_file', default='prediction.json', type=str, help='path for inference')    
    args.add_argument("--threshold", type=float, default=-1, help='entropy threshold to abstrain from answering')
    return args.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.threshold == -1:
        warnings.warn("Threshold value is not set! All predictions are sent to the database.")
    threshold = args.threshold if args.threshold != -1 else float('inf')

    input_file = os.path.join(args.infernece_result_path, args.input_file)
    with open(input_file, 'r') as f:
        data = json.load(f)

    result = {}
    for id_, line in data.items():
        ent = max(line['sequence_entropy'])
        if ent <= threshold:
            pred = line['pred']
        else:
            pred = 'null'
        result[id_] = pred

    out_file = os.path.join(args.infernece_result_path, args.output_file)
    with open(out_file, 'w') as f:
        json.dump(result, f)