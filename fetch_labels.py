import json
import os
from util import *

def fetchLabels(json_file_path):
    data = json.load(open(json_file_path))
    # Fetch unique 'category' values
    unique_categories = set(label['category'] for item in data for label in item.get('labels', []))

    # Print the unique categories
    return sorted(unique_categories)

def writeToFile(output_file_path, categories):
    # Write unique categories to a file line by line
    with open(output_file_path, 'w') as file:
        for category in categories:
            file.write(category + '\n')

    print(f'Unique Categories written to {output_file_path}')



if __name__ == '__main__':
    # folder_path = input('Enter the path to your folder: ')
    JSON_FILE_PATH = '/Users/sanayak/Repositories/AI/object_detection/dataset/BDD_100k/labels/bdd100k_labels_images_val.json'
    OUTPUT_FILE_NAME = 'all_obj.names'
    output_file_path = os.path.join(os.getcwd(), OUTPUT_FILE_NAME)
    labels = fetchLabels(JSON_FILE_PATH)
    print(labels)
    writeToFile(output_file_path, labels)