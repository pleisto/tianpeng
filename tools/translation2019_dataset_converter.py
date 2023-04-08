import argparse
import json
import random

parser = argparse.ArgumentParser(description="Randomly select a percentage of the translation2019 and convert it")
parser.add_argument("input", type=str, help="path to the input JSON file")
parser.add_argument("output", type=str, help="path to the output JSON file")
parser.add_argument("--percentage", type=float, default=0.1, help="percentage of input data to select")

args = parser.parse_args()

with open(args.input, "r") as f:
    data = [json.loads(line) for line in f]
output_data = []

for item in data:
    conversation = "The conversation between human and AI assistant.\n"
    if random.random() < 0.5:
        conversation += f"[|Human|] 翻译成中文: {item['english']}\n"
        conversation += f"[|AI|] {item['chinese']}\n"
    else:
        conversation += f"[|Human|] 翻译成英文: {item['chinese']}\n"
        conversation += f"[|AI|] {item['english']}\n"
    conversation += "[|Human|] "
    output_data.append(conversation)

random_inputs = [i for i in output_data if random.random() < args.percentage]

with open(args.output, "w") as f:
    json.dump(random_inputs, f, indent=2, ensure_ascii=False)
