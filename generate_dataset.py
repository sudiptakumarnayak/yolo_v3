import json
import os
from util import *
from fetch_labels import *

# read the json file to fetch the image and its corresponding label and bounding box co-ordinates.
# skip these categories - [drivable area, lane]

SKIP_CATEGORIES = ['drivable area', 'lane']
label_index_map = {}

JSON_FILE_PATH = '/Users/sanayak/Repositories/AI/object_detection/dataset/BDD_100k/labels/bdd100k_labels_images_val.json'
LABELS_FILE_NAME = 'obj.names'
DATA_DIR = 'data'

DATA_DIR_PATH = os.path.join(os.getcwd(), DATA_DIR)
LABELS_FILE_PATH = os.path.join(DATA_DIR_PATH, LABELS_FILE_NAME)


def create_label_map(file_path):

    with open(file_path, 'r') as file:
        counter = 0
        for index,line in enumerate(file, start=0):
            # Removing leading and trailing whitespaces from the line
            category = line.strip()
            if(category in SKIP_CATEGORIES): # validation check
                continue
            label_index_map[category] = counter
            counter += 1
    
    for key, value in label_index_map.items():
        print(f'{key}: {value}')
    return label_index_map


def process_json(json_file_path):
    data = json.load(open(json_file_path))
    # Create a dictionary of dictionary objects
    class_dict = {}

    for item in data:
        name = item.get('name').strip()
        # name = os.path.splitext(name)[0] # only file name
        labels = item.get('labels', [])
        
        category_dict = {}
        for label in labels:
            category = label.get('category')
            if(category in SKIP_CATEGORIES): # validation check
                continue

            box2d = label.get('box2d', {})
            if box2d.get('x1', 0) >= box2d.get('x2', 0) or box2d.get('y1', 0) >= box2d.get('y2', 0): # validation check : TODO
                    continue
            
            category_dict.setdefault(category, []).append(box2d)

        class_dict[name] = category_dict

    # print(json.dumps(class_dict, indent=2))
    return class_dict


def populate_label_dir(folder_path, class_dict, label_index_map):
    # print(len(class_dict))
    for key in class_dict.keys():
        class_name = os.path.splitext(key)[0]
        file_name = f'{class_name}.txt'
        file_path = os.path.join(folder_path, file_name)
        # print('\n', file_path)

        with open(file_path, 'w') as file:
            category_dict = class_dict[key]
            for category in category_dict.keys():
                label_index = label_index_map.get(category)
                bboxes = category_dict[category]
                for bbox in bboxes:
                    x1 = bbox.get('x1')
                    y1 = bbox.get('y1')
                    x2 = bbox.get('x2')
                    y2 = bbox.get('y2')
                    xcenter, ycenter, w, h = normaliza_bbox(x1, y1, x2, y2)  # NORMALIZING BOUNDING BOX
                    line = ' '.join(map(str, [label_index, xcenter, ycenter, w, h]))
                    # print(line)
                    file.write(line + '\n')
    print(f'Successfully populated labels')


def normaliza_bbox(x1, y1, x2, y2):
    # Darknet label format: [label_index, xcenter, ycenter, w, h] (Relative coordinates)
    W_IMAGE = 1280
    H_IMAGE = 720

    bbox_w = x2 - x1
    bbox_h = y2 - y1
    xcenter = ( x1 + (bbox_w/2) ) / W_IMAGE
    ycenter = ( y1 + (bbox_h/2) ) / H_IMAGE
    w = bbox_w / W_IMAGE
    h = bbox_h / H_IMAGE

    return xcenter, ycenter, w, h

def populate_image_dir(target_dir_path, subset_class_dict):
    for image_name in subset_class_dict.keys():
        image_file_path = subset_class_dict[image_name]
        shutil.copy(image_file_path, target_dir_path)
    print(f'Successfully populated images')


def prepare_val_data():
    # json_file_path = input('Enter the path to your json file: ')
    BDD_JSON_FILE_PATH = '/Users/sanayak/Repositories/AI/object_detection/dataset/BDD_100k/labels/bdd100k_labels_images_val.json'
    BDD_IMG_DIR_PATH = '/Users/sanayak/Repositories/AI/object_detection/dataset/BDD_100k/images/100k/val'

    OUTPUT_FILE_NAME = 'val.txt'
    CHILD_DIR = 'val'
    OUTPUT_FILE_PATH = os.path.join(DATA_DIR_PATH, OUTPUT_FILE_NAME)
    CHILD_DIR_PATH = os.path.join(DATA_DIR_PATH, CHILD_DIR)
    SAMPL_SIZE = 2000

    create_data(BDD_JSON_FILE_PATH, BDD_IMG_DIR_PATH, OUTPUT_FILE_PATH, CHILD_DIR_PATH, SAMPL_SIZE)


def prepare_train_data():
    # json_file_path = input('Enter the path to your json file: ')
    BDD_JSON_FILE_PATH = '/Users/sanayak/Repositories/AI/object_detection/dataset/BDD_100k/labels/bdd100k_labels_images_train.json'
    BDD_IMG_DIR_PATH = '/Users/sanayak/Repositories/AI/object_detection/dataset/BDD_100k/images/100k/train'
    
    OUTPUT_FILE_NAME = 'train.txt'
    CHILD_DIR = 'train'
    OUTPUT_FILE_PATH = os.path.join(DATA_DIR_PATH, OUTPUT_FILE_NAME)
    CHILD_DIR_PATH = os.path.join(DATA_DIR_PATH, CHILD_DIR)
    SAMPL_SIZE = 8000

    create_data(BDD_JSON_FILE_PATH, BDD_IMG_DIR_PATH, OUTPUT_FILE_PATH, CHILD_DIR_PATH, SAMPL_SIZE)


def create_data(BDD_JSON_FILE_PATH, BDD_IMG_DIR_PATH, OUTPUT_FILE_PATH, TARGET_DIR_PATH, SAMPL_SIZE):
    class_dict = process_json(BDD_JSON_FILE_PATH)  # label
    image_dict = get_all_file_names(BDD_IMG_DIR_PATH) # image
    
    # Remove the classes for which image not available
    print('Total num of classes -', len(class_dict.keys()))
    print('Total num of images -', len(image_dict.keys()))
    disjoint_keys_in_class_dict = class_dict.keys() - image_dict.keys()
    print('Keys for which image not available -', len(disjoint_keys_in_class_dict))
    
    # common_keys = class_dict.keys() & image_dict.keys()
    # print('common_keys -', len(common_keys))
    if(len(disjoint_keys_in_class_dict) > 0):
        class_dict = remove_keys(class_dict, disjoint_keys_in_class_dict)
    
    disjoint_keys_in_image_dict = image_dict.keys() - class_dict.keys()
    print('Unlabelled images -', len(disjoint_keys_in_image_dict))
    if(len(disjoint_keys_in_image_dict) > 0):
        image_dict = remove_keys(image_dict, disjoint_keys_in_image_dict)
    

    # create val.txt file - contains path to image file
    subset_image_dict = get_subset(image_dict, SAMPL_SIZE) # subset-images
    # writeToFile(OUTPUT_FILE_PATH, subset_image_dict.values()) # local absolute path

    # print(BDD_IMG_DIR_PATH)
    # print(TARGET_DIR_PATH)
    # Update the absolute path with relative path
    PROJECT_ROOT_PATH = p = os.path.join(os.getcwd(), '')
    CUSTOM_FILE_PATH = str(TARGET_DIR_PATH).removeprefix(str(PROJECT_ROOT_PATH))
    replace_portion = lambda path: path.replace(str(BDD_IMG_DIR_PATH), str(CUSTOM_FILE_PATH))
    modified_image_dict = {key: replace_portion(value) for key, value in subset_image_dict.items()}
    writeToFile(OUTPUT_FILE_PATH, modified_image_dict.values()) # relative absolute path


    #  DATASET CREATION
    #  create a 'val' folder (if nt exists)
    create_dir(TARGET_DIR_PATH)
    # fetch the respective labels for subset_image_dict
    subset_class_dict = fetch_elements_by_keys(class_dict, subset_image_dict.keys()) # subset-class

    # re-create val folder 
    print('TARGET_DIR_PATH -', TARGET_DIR_PATH)
    recreate_dir(TARGET_DIR_PATH)
    populate_label_dir(TARGET_DIR_PATH, subset_class_dict, label_index_map)
    populate_image_dir(TARGET_DIR_PATH, subset_image_dict)
    print('Num of files - ', len(os.listdir(TARGET_DIR_PATH)))



def populate_labels():
    labels = fetchLabels(JSON_FILE_PATH)
    labels = [item for item in labels if item not in SKIP_CATEGORIES]
    # print(LABELS_FILE_PATH)
    writeToFile(LABELS_FILE_PATH, labels)
    

    






if __name__ == '__main__':
    
    # To fetch all the labels run the fetch_labels.py

    create_dir('data')

    populate_labels()
    print('Successfully populated labels \n')

    label_index_map = create_label_map(LABELS_FILE_PATH)
    print('Successfully created label-index map \n')

    prepare_val_data()
    print('Successfully populated val dataset \n')
    
    prepare_train_data()
    print('Successfully populated train dataset \n')


