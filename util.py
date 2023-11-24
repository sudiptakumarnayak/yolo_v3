import os
import random
import shutil

SUB_DIR_NAMES = ['train', 'val']


def get_all_file_names(folder_path):
    # Get all file names in the folder
    file_names = os.listdir(folder_path)

    # Create a dictionary to store the file names with keys as filenames (excluding extensions)
    # file_dict = {os.path.splitext(file_name)[0]: '' for file_name in file_names} # Fetch only image id
    file_dict = {file_name: os.path.join(folder_path, file_name) for file_name in file_names}

    return file_dict


def writeToFile(output_file_path, lines):
    # Write unique categories to a file line by line
    with open(output_file_path, 'w') as file:
        for line in lines:
            file.write(line + '\n')

    print(f'Successfully written to {output_file_path} \n')


def remove_dir(directory_path):
     if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            # print(f'Folder {directory_path} has been deleted.')
        except OSError as e:
            print(f'Error: {e}')


def recreate_dir(directory_path):
    remove_dir(directory_path)
    os.makedirs(directory_path)


def create_dir(dir):
    os.makedirs(dir, exist_ok=True) # No error thrown if the folders already exist


def remove_keys(original_dict, keys_to_remove):
    updated_dict = {key: value for key, value in original_dict.items() if key not in keys_to_remove}
    return updated_dict


def get_subset(original_dict, num_items):
    subset_keys = random.sample(original_dict.keys(), num_items) # pick random images
    # subset_keys = list(original_dict.keys())[:num_items]
    subset_dict = {key: original_dict[key] for key in subset_keys}
    return subset_dict


def fetch_elements_by_keys(original_dict, keys_to_fetch):
    fetched_elements = {key: original_dict[key] for key in keys_to_fetch if key in original_dict}
    return fetched_elements
