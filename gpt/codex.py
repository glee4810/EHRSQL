import os
import json
import time
import openai
import argparse
import numpy as np
from tqdm import tqdm


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--api_key_path', default='OPENAI_API_KEY.json', type=str, help='path for openai api key')
    args.add_argument('--prompt_path', default='', type=str, help='path for prompt')
    args.add_argument('--test_data_path', required=True, type=str, help='eval data path')
    args.add_argument('--infernece_result_path', default='./', type=str, help='path for inference')
    args.add_argument('--output_file', default='prediction.json', type=str, help='outnput file name')
    return args.parse_args()

def run_engine(prompt):
    response = openai.Completion.create(
      engine="code-davinci-002",
      prompt=prompt,
      temperature=0,
      max_tokens=512,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=["#", ";"]
    )
    text = response['choices'][0]['text']
    text = f'select{text}'
    return text

if __name__ == '__main__':
    args = parse_args()

    with open(args.api_key_path) as f:
        OPENAI_API_KEY = json.load(f)
    openai.api_key = OPENAI_API_KEY["API_KEY"]

    if args.prompt_path == '':
        prompt = ''
    else:
        with open(args.prompt_path) as f:
            prompt = f.read()

    with open(args.test_data_path) as json_file:
        data = json.load(json_file)

    result = {}
    for line in tqdm(data):
        id_ = line['id']
        question = line['question']
        prompt_to_run = prompt
        while True:
            try:
                prompt_to_run = prompt_to_run.replace('TEST_QUESTION', question)
                pred = run_engine(prompt_to_run)
                break
            except KeyboardInterrupt:
                exit()
            except:
                time.sleep(60)
        result[id_] = pred

    os.makedirs(args.infernece_result_path, exist_ok=True)
    out_file = os.path.join(args.infernece_result_path, args.output_file)
    with open(out_file, 'w') as f:
        json.dump(result, f)