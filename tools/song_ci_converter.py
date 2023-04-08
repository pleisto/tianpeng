import argparse
import json

parser = argparse.ArgumentParser(description="Randomly select a percentage of the song ci dataset and convert it")
parser.add_argument("input", type=str, help="path to the input JSON file")
parser.add_argument("output", type=str, help="path to the output JSON file")

args = parser.parse_args()

with open(args.input, "r") as f:
    data = json.load(f)

inputs = [f"宋词《{d['rhythmic']}》\n 作者: {d['author']}\n{''.join(d['paragraphs'])}\n" for d in data]


with open(args.output, "w") as f:
    json.dump(inputs, f, indent=2, ensure_ascii=False)
