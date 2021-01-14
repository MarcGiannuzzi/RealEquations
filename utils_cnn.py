import tensorflow as tf
import os
import os.path
import pickle
from tqdm import tqdm
import cv2 as cv
import numpy as np
import random
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

from utils_writing import *

MATH_SIGNS = ['-', '+', '=', 'times', 'sum', 'div', '(', ')']
MATH_LETTERS = ['X']
MATH_DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
TOTAL_MATH_SYMBOLS = MATH_SIGNS + MATH_LETTERS + MATH_DIGITS


TOTAL_NUMBER_OF_FEATURES = len(MATH_SIGNS) + len(MATH_LETTERS) + len(MATH_DIGITS)
PIXEL_SIZE_IMAGES = (45, 45)
INPUT_SHAPE_CNN = (45, 45, 1)
FIT_NB_EPOCHS = 20
#Important : It is a classification problem. Final output vector will be of size len(MATH_SIGNS) + len(MATH_LETTERS) + len(MATH_DIGITS)
# In the output vector, letters indices are greater than the signs indices that are greater than the digits indices
# [letters, signs, digits]

#Important : It is a classification problem. Final output vector will be of size len(MATH_SIGNS) + len(MATH_LETTERS) + len(MATH_DIGITS)
# In the output vector, letters indices are greater than the signs indices that are greater than the digits indices
# [letters, signs, digits]



def load_signs_data():
    all_signs_data = []
    for math_sign_number, math_sign in enumerate(MATH_SIGNS):
        print("Loading files from ", math_sign, ' sign : (', math_sign_number + 1, ' | ', len(MATH_SIGNS), ')')
        
        print("Loading signs files from crohme dataset")    
        crohme_directory_math_sign = './data/crohme/labellized_data/' + math_sign
        crohme_sign_image_files = os.listdir(crohme_directory_math_sign)
        number_crohme_sign_image_files = 0
        for crohme_sign_image_file in tqdm(crohme_sign_image_files):
            relative_path_to_crohme_sign_file = crohme_directory_math_sign + '/' + crohme_sign_image_file
            crohme_sign_image = cv.imread(relative_path_to_crohme_sign_file, cv.IMREAD_GRAYSCALE)
            all_signs_data.append([crohme_sign_image, math_sign_number])
            number_crohme_sign_image_files += 1
        
        print("Loading signs files from kaggle dataset")    
        kaggle_directory_math_sign = './data/kaggle_math_images_data/' + math_sign
        kaggle_sign_image_files = os.listdir(kaggle_directory_math_sign)
        number_kaggle_sign_image_files = 0
        for kaggle_sign_image_file in tqdm(kaggle_sign_image_files):
            if number_kaggle_sign_image_files < number_crohme_sign_image_files:
                relative_path_to_kaggle_sign_file = kaggle_directory_math_sign + '/' + kaggle_sign_image_file
                kaggle_sign_image = cv.imread(relative_path_to_kaggle_sign_file, cv.IMREAD_GRAYSCALE)
                all_signs_data.append([kaggle_sign_image, math_sign_number])
                number_kaggle_sign_image_files += 1
            else:
                print('Stop fetching in order to have balanced data between sources')
                break
            
            
    pickle.dump(all_signs_data, open("./data/signs_data.p", "wb" ))
    print("Signs data successfully loaded\n")
    return all_signs_data
        
def load_letters_data():
    all_letters_data = []
    for math_letter_number, math_letter in enumerate(MATH_LETTERS):
        print("Loading files from ", math_letter, ' letter : (', math_letter_number + 1, ' | ', len(MATH_LETTERS), ')')
        
        print("Loading letters files from crohme dataset")    
        crohme_directory_math_letter = './data/crohme/labellized_data/' + 'x'
        crohme_letter_image_files = os.listdir(crohme_directory_math_letter)
        number_crohme_letter_image_files = 0
        for crohme_letter_image_file in tqdm(crohme_letter_image_files):
            relative_path_to_crohme_letter_file = crohme_directory_math_letter + '/' + crohme_letter_image_file
            crohme_letter_image = cv.imread(relative_path_to_crohme_letter_file, cv.IMREAD_GRAYSCALE)
            all_letters_data.append([crohme_letter_image, len(MATH_LETTERS) + math_letter_number])
            number_crohme_letter_image_files += 1
        
        print("Loading letters files from kaggle dataset")    
        kaggle_directory_math_letter = './data/kaggle_math_images_data/' + math_letter
        kaggle_letter_image_files = os.listdir(kaggle_directory_math_letter)
        number_kaggle_letter_image_files = 0
        for kaggle_letter_image_file in tqdm(kaggle_letter_image_files):
            if number_kaggle_letter_image_files < number_crohme_letter_image_files:
                relative_path_to_kaggle_letter_file = kaggle_directory_math_letter + '/' + kaggle_letter_image_file
                kaggle_letter_image = cv.imread(relative_path_to_kaggle_letter_file, cv.IMREAD_GRAYSCALE)
                all_letters_data.append([kaggle_letter_image, len(MATH_SIGNS) + math_letter_number])
                number_kaggle_letter_image_files += 1
            else:
                print('Stop fetching in order to have balanced data between sources')
                break
            
            
    pickle.dump(all_letters_data, open("./data/letters_data.p", "wb" ))
    print("Letters data successfully loaded\n")
    return all_letters_data


def load_digits_data():
    all_digits_data = []
    for math_digit_number, math_digit in enumerate(MATH_DIGITS):
        print("Loading files from ", math_digit, ' digit : (', math_digit_number + 1, ' | ', len(MATH_DIGITS), ')')
        
        print("Loading digits files from crohme dataset")    
        crohme_directory_math_digit = './data/crohme/labellized_data/' + math_digit
        crohme_digit_image_files = os.listdir(crohme_directory_math_digit)
        number_crohme_digit_image_files = 0
        for crohme_digit_image_file in tqdm(crohme_digit_image_files):
            relative_path_to_crohme_digit_file = crohme_directory_math_digit + '/' + crohme_digit_image_file
            crohme_digit_image = cv.imread(relative_path_to_crohme_digit_file, cv.IMREAD_GRAYSCALE)
            all_digits_data.append([crohme_digit_image, len(MATH_SIGNS) + len(MATH_LETTERS) + math_digit_number])
            number_crohme_digit_image_files += 1
        
        print("Loading digits files from kaggle dataset")
        kaggle_directory_math_digit = './data/kaggle_math_images_data/' + math_digit
        kaggle_digit_image_files = os.listdir(kaggle_directory_math_digit)
        number_kaggle_digit_image_files = 0
        for kaggle_digit_image_file in tqdm(kaggle_digit_image_files):
            if number_kaggle_digit_image_files < number_crohme_digit_image_files:
                relative_path_to_kaggle_digit_file = kaggle_directory_math_digit + '/' + kaggle_digit_image_file
                kaggle_digit_image = cv.imread(relative_path_to_kaggle_digit_file, cv.IMREAD_GRAYSCALE)
                all_digits_data.append([kaggle_digit_image, len(MATH_SIGNS) + len(MATH_LETTERS) + math_digit_number])
                number_kaggle_digit_image_files += 1
            else:
                print('Stop fetching in order to have balanced data between sources')
                break
            
            
    pickle.dump(all_digits_data, open("./data/digits_data.p", "wb" ))
    print("Digits data successfully loaded\n")
    return all_digits_data
    
    
def load_all_data():
    print("Loading data...")
    all_data = None
    if not os.path.exists("./data/all_data.p"):
        if not os.path.exists("./data/signs_data.p"):
            load_signs_data()
        if not os.path.exists("./data/letters_data.p"):
            load_letters_data()
        if not os.path.exists("./data/digits_data.p"):
            load_digits_data()


        all_signs_data = pickle.load(open("./data/signs_data.p", "rb"))
        all_letters_data = pickle.load(open("./data/letters_data.p", "rb"))
        all_digits_data = pickle.load(open("./data/digits_data.p", "rb"))

        all_data = all_signs_data + all_letters_data + all_digits_data
        pickle.dump(all_data, open( "./data/all_data.p", "wb" ))
        
    else:
        all_data = pickle.load(open("./data/all_data.p", "rb"))
    return all_data
        
        

def balance_data(all_data):
    all_data_balanced = []
    classes_data = [[] for i in range(TOTAL_NUMBER_OF_FEATURES)]
    
    
    for data_point in all_data:
        data_point_class = data_point[1]
        image_data_point = data_point[0]
        classes_data[data_point_class].append(data_point)
        
        
    lengths_classes_data = [len(class_data) for class_data in classes_data]
    min_length_class_data = min(lengths_classes_data)
    
    for index_class_data in range(len(classes_data)):
        reduced_class_data = classes_data[index_class_data][:min_length_class_data - 1]
        classes_data[index_class_data] = reduced_class_data
        
    for class_data in classes_data:
        for data_point in class_data:
            all_data_balanced.append(data_point)
    
    return all_data_balanced, min_length_class_data
    
        
        
def shuffle_data(data):
    random.shuffle(data)        


def create_train_test_data(all_data, split_percentage):
    number_of_images = len(all_data)
    split_index = int(split_percentage * number_of_images)
    
    train_data = all_data[:split_index]
    test_data = all_data[split_index + 1:]
    
    train_images = np.array([np.array(train_data_point[0], dtype='float64') / 255 for train_data_point in train_data])
    train_labels = np.array([train_data_point[1] for train_data_point in train_data])
    test_images = np.array([np.array(test_data_point[0], dtype='float64') / 255 for test_data_point in test_data])
    test_labels = np.array([test_data_point[1] for test_data_point in test_data])
    
    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)


    
    return (train_images, train_labels), (test_images, test_labels)



def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE_CNN))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(TOTAL_NUMBER_OF_FEATURES, activation='softmax'))
    return model

def compile_model(model, loss):
    metrics = ['accuracy']
    model.compile(optimizer='adam',
              loss=loss,
              metrics=metrics)
    
    
    
def fit_model(model, train_images, train_labels, test_images, test_labels):
    history = model.fit(train_images, train_labels, epochs=FIT_NB_EPOCHS, 
                        validation_data=(test_images, test_labels))
    return history


def evaluate_model(model, history):
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    
    
    
def load_model(model_name, loss=None, train_images=None, train_labels=None, test_images=None, test_labels=None):
    print("Loading model")
    model = None
    returned_history = None
    
    if not os.path.exists("./" + model_name):
        os.mkdir("./" + model_name)
    
    if not os.path.exists("./" + model_name + "/saved_model.pb"):
        model = create_model()
        compile_model(model, loss)
        returned_history = fit_model(model, train_images, train_labels, test_images, test_labels)
        model.save(".\\" + model_name)
        pickle.dump(returned_history.history, open("./" + model_name + "/model_history.json", "wb" ))
        
        
    else:
        model = models.load_model(".\\" + model_name)
        returned_history = pickle.load(open("./" + model_name + "/model_history.json", "rb"))

    print("\nModel loaded.")
    return model, returned_history
