import argparse
import json
import random

parser = argparse.ArgumentParser(description="Randomly select a percentage of the baize dataset and convert it")
parser.add_argument("input", type=str, help="path to the input JSON file")
parser.add_argument("output", type=str, help="path to the output JSON file")
parser.add_argument("--percentage", type=float, default=0.1, help="percentage of input data to select")

args = parser.parse_args()

with open(args.input, "r") as f:
    data = json.load(f)

inputs = [d["input"] for d in data]

random_inputs = [i for i in inputs if random.random() < args.percentage]

with open(args.output, "w") as f:
    json.dump(random_inputs, f, indent=2, ensure_ascii=False)
