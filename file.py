import os
import re

# Define the root directory where your files are located
root_directory = "runs/pneumoniamnist/equal-jempp-adam/checkpoints"

# Iterate over each directory and its contents using os.walk
for dirpath, dirnames, filenames in os.walk(root_directory):
    for filename in filenames:
        if filename.endswith(".ckpt"):
            # Construct the full path to the checkpoint file
            checkpoint_path = os.path.join(dirpath, filename)

            # Extract the number of labels from the parent folder name
            parent_folder = os.path.basename(os.path.dirname(checkpoint_path))
            match = re.search(r"(\d+)", parent_folder)
            if match:
                num_labels = match.group(1)
            else:
                raise ValueError("Number of labels not found in folder name")

            # ckpt type is best if it contains the word epoch
            if "epoch" in checkpoint_path:
                ckpt_type = "best"
            else:
                ckpt_type = "last"

            # Define the new filename
            new_filename = f"num-labels-{num_labels}_{ckpt_type}.ckpt"

            # Generate the new path
            new_path = os.path.join(root_directory, new_filename)

            print(f"Renaming '{checkpoint_path}' to '{new_path}'")

            #
            # Rename the checkpoint file
            os.rename(checkpoint_path, new_path)
            # print(f"Renamed '{checkpoint_path}' to '{new_path}'")
            # remove empty folders in root_directory

            # # Optionally, if you want to move the file one layer out:
            # destination_directory = parent_directory
            # os.rename(new_path, os.path.join(destination_directory, new_filename))
            # print(f"Moved '{new_path}' to '{destination_directory}'")
