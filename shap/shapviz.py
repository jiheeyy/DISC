# IMPORT
# cv2, matplotlib, pandas, numpy, random, shap, tensorflow
import datetime

import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import shap
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

#################################################

truemean = pd.read_csv('attribute_means.csv')


def true_mean(stim):
    # For i+1.jpg, should look for truemean['happy'][i]
    stim = int(stim)
    return truemean["happy"][stim - 1]

print('csv and packages done')
##################################################

# Load the ResNet50 model with given input shape and without the top (fully-connected) layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(1024, 1024, 3))

# Freeze all the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add a global spatial average pooling layer
x = GlobalAveragePooling2D()(base_model.output)

# Add a fully-connected layer with 1 neuron
predictions = Dense(1)(x)

# Construct the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Load the weights for the last two layers from the HDF5 file
model.load_weights('checkpoints/wrwf.hdf5', by_name=True)

print('model weight load done')
##################################################
def preprocess_image(image_path, target_size=(1024, 1024)):
    # Load the image
    img = image.load_img(image_path, target_size=target_size)

    # Convert image to a numpy array
    img_array = image.img_to_array(img)

    # Expand the dimensions to match the expected input format of the model
    # i.e., (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image if required (e.g., if using ResNet50's pre-trained weights)
    img_array /= 255.0

    return img_array


def show_image(img_array):
    # Since we expanded dimensions for model input, we need to squeeze
    # the first dimension to visualize it.
    # The shape will change from (1, 1024, 1024, 3) to (1024, 1024, 3)
    img_array = np.squeeze(img_array, axis=0)

    plt.imshow(img_array)
    plt.axis('off')  # To hide the axis values
    plt.show()


# Visualize the processed_image
# image_path = "test_set/1004.jpg"
# prim = preprocess_image(image_path)
# show_image(prim)
# model.predict(prim)


##################################################

def f(X):
    tmp = X.copy()
    preprocess_input(tmp)
    return model(tmp)


def create_X():
    path = os.path.join('test_set/')
    X = []
    stimulus = []
    i = 0
    for img in os.listdir(path):
        if i >= 50:
            break
        else:
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (1024, 1024))
                X.append(new_array)  # Directly append the resized image to X
                stimulus.append(img[:-4])
                i += 1
            except Exception as e:
                print(f"Error processing image: {img}. Error: {e}")
    return np.array(X), stimulus  # Convert the list to a numpy array


X, stimulus = create_X()

print('created X, stimulus done')

# wrwf
# define a masker that is used to mask out partitions of the input image, this one uses a blurred background
masker = shap.maskers.Image("inpaint_telea", X[0].shape)

explainer = shap.Explainer(f, masker)  # "output_names=class_names"

# here we use 500 evaluations of the underlying model to estimate the SHAP values
img_num = 0
shap_values = explainer(X[img_num:img_num+1], max_evals=500, batch_size=25, outputs=shap.Explanation.argsort.flip[:1])
print('shap value calculation done')
shap.image_plot(shap_values, show=False)
print('shap plot done')
if not os.path.exists('results'):
    print('NO RESULTS! MAKING NEW FOLDER')
    os.makedirs('results')
now = datetime.now()
formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
formatted_date = formatted_date.replace(':', '_').replace(' ', '_')
plt.savefig('results/stimulus_' + str(stimulus[img_num]) + '_' + str(formatted_date) + '.png')
print('shap plot save done')

print(f"STIMULUS NUM: {stimulus[img_num]}")
print(true_mean(stimulus[img_num]))
print(model.predict(preprocess_image("test_set/"+str(stimulus[img_num])+".jpg")))
print(shap_values.base_values)
