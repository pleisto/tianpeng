import json
import os

input_directory = "dataset"
output_file = "all.json"


def merge_json_files(input_directory, output_file):
    all_data = []

    for filename in os.listdir(input_directory):
        if filename.endswith(".json") and filename != output_file:
            file_path = os.path.join(input_directory, filename)
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error decoding {os.path.basename(file_path)}: {e}")
                all_data.extend(data)

    with open(os.path.join(input_directory, output_file), "w") as outfile:
        json.dump(all_data, outfile, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    merge_json_files(input_directory, output_file)
