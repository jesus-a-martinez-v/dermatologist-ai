from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras import metrics
import os
import glob
import matplotlib.image as mpimg
import numpy as np
from keras.utils.np_utils import to_categorical
from pathlib import Path

import cv2

BASE_PATH = os.path.join("/", "floyd", "input", "data")
TRAIN_DATA = os.path.join(BASE_PATH, "train")
VALIDATION_DATA = os.path.join(BASE_PATH, "valid")
TEST_DATA = os.path.join(BASE_PATH, "test")

MAPPING = {'seborrheic_keratosis': 0, 
           'melanoma': 1, 
           'nevus': 2}

def load_dataset(image_directory, dataset):
    image_list = []
    image_labels = []
    image_types = {'seborrheic_keratosis', 'melanoma', 'nevus'}
    
    x_name = "./%s_images.npy" % dataset
    y_name = "./%s_labels.npy" % dataset
    
    # Iterave over each subfolder corresponding to the type of image and add the image to the resulting list.
    if not Path(x_name).is_file() or not Path(y_name).is_file():
        for image_type in image_types:
            print("Loading images in folder: %s" % os.path.join(image_directory, image_type))
        
            for file in glob.glob(os.path.join(image_directory, image_type, '*')):
                image = mpimg.imread(file)
                print(image.shape)
                image = cv2.resize(image, (299, 299))
                print(image.shape)
            
                if image is not None:
                    image_list.append(image)
                    image_labels.append(MAPPING[image_type])
                
        image_list = np.array(image_list)
        image_labes = np.array(image_labels)
    
        np.save(x_name, image_list)
        np.save(y_name, image_labels)
    else:
        image_list = np.load(x_name)
        image_labels = np.load(y_name)
    
    return image_list, image_labels


# TODO Create a function for each pretrained model.
def get_model():
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(3, activation="softmax")(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
        
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc', metrics.categorical_accuracy])
    
    return model


def train_model(model, 
                epochs=50, 
                image_size=(299, 299, 3), 
                batch_size=64, 
                train_steps_per_epoch=2000,
                validation_steps=800):
    
    # Data agumentation generators
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Actual generators
    x_train, y_train = load_dataset(TRAIN_DATA, "train")
    y_train = to_categorical(y_train, num_classes=3)
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    
    x_validation, y_validation = load_dataset(VALIDATION_DATA, "validation")
    y_validation = to_categorical(y_validation, num_classes=3)
    validation_generator = test_datagen.flow(x_validation, y_validation, batch_size=batch_size)
    
    callbacks = [
        TensorBoard(),
        EarlyStopping(patience=3),
        ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    ]
    
    model.fit_generator(train_generator,
                        # steps_per_epoch=train_steps_per_epoch,
                        epochs=epochs, 
                        validation_data=validation_generator, 
                        # validation_steps=validation_steps, 
                        callbacks=callbacks)
    
    
if __name__ == "__main__":
    m = get_model()
    train_model(m, batch_size=256)
    