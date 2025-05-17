import json
import os

def merge_json_files(filenames, output_path):
    """Merge multiple JSON files into one."""
    merged_data = []
    for file_path in filenames:
        with open(file_path, 'r', encoding='utf-8') as infile:
            merged_data.extend(json.load(infile))
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(merged_data, outfile, indent=2, ensure_ascii=False)
    
    print(f"Merged {len(filenames)} files into {output_path}")

def convert_to_instruction_format(input_path, output_path):
    """Convert Q&A format to instruction format for fine-tuning."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    instruction_data = []
    for item in data:
        question = item.get("title", "").strip()
        answer = item.get("best_answer", {}).get("body", "").strip()

        if question and answer:
            instruction_data.append({
                "instruction": question,
                "input": "",
                "output": answer
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(instruction_data, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(instruction_data)} entries to instruction format at {output_path}")

# 📝 List of JSON files to merge
input_files = [
    
]

# ✅ Provide correct file paths
merged_output = "qa_combined.json"
formatted_output = "qa_formatted.json"

# 🔁 Execute
merge_json_files(input_files, merged_output)
convert_to_instruction_format(merged_output, formatted_output)
