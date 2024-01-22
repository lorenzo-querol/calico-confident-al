import os

# get the current working directory
cwd = "runs/bloodmnist_epoch_150_sgd/checkpoints/"

# get the list of folders in the current working directory
folders = os.listdir(cwd)

for folder in folders:
    # check if the folder name contains the string 'active'
    if "active" in folder:
        # my folder name is in the format 'active_NUMBER' I want to insert the word 'calibrated between active and NUMBER
        # so I split the string at the underscore and insert the word calibrated
        new_folder = folder.split("_")
        new_folder.insert(1, "calibrated")
        new_folder = "_".join(new_folder)
        # create the new folder name
        new_folder = cwd + new_folder
        # create the old folder name
        folder = cwd + folder
        # rename the folder
        os.rename(folder, new_folder)
        # print the new folder name
        print(new_folder)

if "samples" not in cwd:
    for folder in folders:
        # check if the folder name contains the string 'active'
        if "baseline" in folder:
            # my folder name is in the format 'baseline_NUMBER' I want to replace baseline with active and insert the word uncalibrated between active and NUMBER
            # so I split the string at the underscore and insert the word calibrated
            new_folder = folder.split("_")
            new_folder[0] = "active"
            new_folder.insert(1, "uncalibrated")
            new_folder = "_".join(new_folder)
            # create the new folder name
            new_folder = cwd + new_folder
            # create the old folder name
            folder = cwd + folder
            # rename the folder
            os.rename(folder, new_folder)
            # print the new folder name
            print(new_folder)
