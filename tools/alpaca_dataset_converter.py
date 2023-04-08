import argparse
import json
import random

parser = argparse.ArgumentParser(description="Randomly select a percentage of the Alpaca dataset and convert it")
parser.add_argument("input", type=str, help="path to the input JSON file")
parser.add_argument("output", type=str, help="path to the output JSON file")
parser.add_argument("--percentage", type=float, default=0.1, help="percentage of input data to select")
parser.add_argument("--jsonl", type=bool, default=False, help="whether the input file is in JSONL format")

args = parser.parse_args()

with open(args.input, "r") as f:
    data = json.load(f) if not args.jsonl else [json.loads(line) for line in f]
output_data = []

for item in data:
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output_text = item.get("output", "")
    conversation = "The conversation between human and AI assistant.\n"
    if instruction:
        conversation += f"[|Human|] {instruction}\n"
    if input_text:
        conversation += f"{input_text}\n"
    if output_text:
        conversation += f"[|AI|] {output_text}\n"
    conversation += "[|Human|] "
    output_data.append(conversation)

random_inputs = [i for i in output_data if random.random() < args.percentage]

with open(args.output, "w") as f:
    json.dump(random_inputs, f, indent=2, ensure_ascii=False)
