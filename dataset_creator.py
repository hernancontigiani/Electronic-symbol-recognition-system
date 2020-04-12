#!/usr/bin/env python
"""Dataset creator

This script takes the original images of electronic components and
make a dataset from the variation of rotation and offset from each symbol class.

Finally it generates the train and validation dataset
The test dataset is created by hand modification in order to have new images to test the model

IMPORTANT: This script could takes 30 minutes to complete

"""

import os
import sys
import shutil
from PIL import Image
from PIL import ImageChops
from sklearn.model_selection import train_test_split

__author__ = "Hernan Contigiani"
__email__ = "hernan4790@gmail.com"
__version__ = "1.0.0"

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def create_dataset_from_original (reference_image, output_path):
    
    i = 2
    # setup toolbar
    toolbar_width = 40
    step = ((300/30) * (300/30) * (180/2) / toolbar_width)
    printProgressBar(0, toolbar_width, prefix = 'Progress:', suffix = 'Complete', length = toolbar_width, fill = '>')

    for offset_value_x in range (-150,150,30):

        for offset_value_y in range (-150,150,30):

            offset_img = ImageChops.offset(reference_image,offset_value_x,offset_value_y)
            for rot_value in range (0,180,2):
                new_img = offset_img.rotate(-rot_value)
                new_img.save(output_path + str(i) + ".png", "PNG")
                i = i+1

                if (i % int(step)) == 0:
                    printProgressBar(i/step, toolbar_width, prefix = 'Progress:', suffix = 'Complete', length = toolbar_width, fill = '>')

def train_valid_dataset (dataset_path, trainset_path, validset_path, valid_size = 0.2):
    
    # Read all images in dataset_path
    symbols = os.listdir(dataset_path)
    symbols.sort()
    
    # Generate a train and valid set from the original dataset
    trainset, validset = train_test_split(symbols, test_size=valid_size)

    # Generate image paths for training data
    src_data = [dataset_path+'/{}'.format(image) for image in trainset]
    dest_data = [trainset_path+'/{}'.format(image) for image in trainset]

    # Copy from dataset folder to train folder
    for i in range(len(src_data)):
        shutil.copy(src_data[i], dest_data[i])


    # Generate image paths for valid data
    src_data = [dataset_path+'/{}'.format(image) for image in validset]
    dest_data = [validset_path+'/{}'.format(image) for image in validset]

    # Copy from dataset folder to valid folder
    for i in range(len(src_data)):
        shutil.copy(src_data[i], dest_data[i])


def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)

def main():

    directory = "./symbols"

    if os.access(directory, os.F_OK) == True:
        shutil.rmtree(directory, onerror=remove_readonly)


    os.mkdir(directory)
    if os.access(directory, os.F_OK) == False:
        print("Error: Directory", symbols, "could not be created!")
        quit()

    capacitor_path = directory+"/capacitor"
    inductor_path = directory+"/inductor"
    resistance_path = directory+"/resistance"

    os.mkdir(capacitor_path)
    os.mkdir(inductor_path)
    os.mkdir(resistance_path)

    reference_image_name = "L1.png"
    output_path =  inductor_path + "/L"
    reference_image = Image.open(reference_image_name)
    print("Creating inductor symbols...")
    create_dataset_from_original (reference_image, output_path)

    reference_image_name = "R1.png"
    output_path =  resistance_path + "/R"
    reference_image = Image.open(reference_image_name)
    print("Creating resistance symbols...")
    create_dataset_from_original (reference_image, output_path)

    reference_image_name = "C1.png"
    output_path =  capacitor_path + "/C"
    print("Creating capacitor symbols...")
    reference_image = Image.open(reference_image_name)
    create_dataset_from_original (reference_image, output_path)

    dataset_path = "./dataset"

    capacitor_train_path = dataset_path+"/train/capacitor"
    capacitor_valid_path = dataset_path+"/valid/capacitor"
    inductor_train_path = dataset_path+"/train/inductor"
    inductor_valid_path = dataset_path+"/valid/inductor"
    resistance_train_path = dataset_path+"/train/resistance"
    resistance_valid_path = dataset_path+"/valid/resistance"

    if os.access(dataset_path, os.F_OK) == True:
        shutil.rmtree(dataset_path, onerror=remove_readonly)


    os.mkdir(dataset_path)
    if os.access(dataset_path, os.F_OK) == False:
        print("Error: Directory", dataset_path, "could not be created!")
        quit()

    os.mkdir(dataset_path+'/train')
    os.mkdir(dataset_path+'/valid')

    os.mkdir(capacitor_train_path)
    os.mkdir(capacitor_valid_path)
    os.mkdir(inductor_train_path)
    os.mkdir(inductor_valid_path)
    os.mkdir(resistance_train_path)
    os.mkdir(resistance_valid_path)
    
    print("Creating dataset directory...")
    train_valid_dataset(dataset_path=capacitor_path, trainset_path=capacitor_train_path, validset_path=capacitor_valid_path)
    train_valid_dataset(dataset_path=inductor_path, trainset_path=inductor_train_path, validset_path=inductor_valid_path)
    train_valid_dataset(dataset_path=resistance_path, trainset_path=resistance_train_path, validset_path=resistance_valid_path)

    shutil.copytree('./testset', dataset_path+'/test')


if __name__ == '__main__':
    main()
    
