from tqdm import tqdm
import os
import lzma

def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

# Define paths
folder_path = "C:/Users/darsh/Documents/GPT/OpenWebText/subsets/urlsf_subset00/openwebtext"
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"

# Get list of .xz files
files = xz_files_in_dir(folder_path)
total_files = len(files)

# Calculate the split index (90% for training)
split_index = int(total_files * 0.9)
files_train = files[:split_index]
files_val = files[split_index:]

# Set to store vocabulary
vocab = set()

# Process the training files
with open(output_file_train, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_train, total=len(files_train), desc="Processing training files"):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

# Process the validation files
with open(output_file_val, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_val, total=len(files_val), desc="Processing validation files"):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

# Write the vocabulary to vocab.txt
with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in vocab:
        vfile.write(char + '\n')

print(f"Process completed. Training and validation files created with the combined vocabulary in {vocab_file}.")
