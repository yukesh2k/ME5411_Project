import os
import shutil
import random

# Get the current directory and the target directory
character = "H"
current_directory = f"./train/{character}"
target_directory = f'./test/{character}/'  # Change this to the desired target path

# List all files in the current directory
all_files = [f for f in os.listdir(current_directory) if os.path.isfile(os.path.join(current_directory, f))]

# Calculate 25% of the total files
num_files_to_move = int(len(all_files) * 0.25)

# Randomly select 25% of the files
files_to_move = random.sample(all_files, num_files_to_move)

# Move selected files to the target directory
for file in files_to_move:
    source = os.path.join(current_directory, file)
    destination = os.path.join(target_directory, file)
    
    try:
        shutil.move(source, destination)
        print(f"Moved: {file}")
    except Exception as e:
        print(f"Error moving {file}: {e}")
