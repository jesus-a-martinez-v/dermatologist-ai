import glob
import os
from pathlib import Path

import cv2
import matplotlib.image as mpimg
import numpy as np
from keras import metrics
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.engine.saving import load_model
from keras.layers import Dense, GlobalAveragePooling2D, Activation, BatchNormalization, LeakyReLU
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

BASE_PATH = os.path.join('.', 'data')

TRAIN_DATA = os.path.join(BASE_PATH, 'train')
VALIDATION_DATA = os.path.join(BASE_PATH, 'valid')
TEST_DATA = os.path.join(BASE_PATH, 'test')

MAPPING = {'seborrheic_keratosis': 0,
           'melanoma': 1,
           'nevus': 2}


def load_dataset(image_directory, dataset):
    image_list = []
    image_labels = []
    image_types = {'seborrheic_keratosis', 'melanoma', 'nevus'}

    x_name = f'./{dataset}_images.npy'
    y_name = f'./{dataset}_labels.npy'

    # Iterate over each subfolder corresponding to the type of image and add the image to the resulting list.
    if not Path(x_name).is_file() or not Path(y_name).is_file():
        for image_type in image_types:
            print('Loading images in folder: {os.path.join(image_directory, image_type)}')

            for file in glob.glob(os.path.join(image_directory, image_type, '*')):
                image = mpimg.imread(file)
                print(image.shape)
                image = cv2.resize(image, (299, 299))
                print(image.shape)

                if image is not None:
                    image_list.append(image)
                    image_labels.append(MAPPING[image_type])

        image_list = np.array(image_list)
        image_labels = np.array(image_labels)

        np.save(x_name, image_list)
        np.save(y_name, image_labels)
    else:
        image_list = np.load(x_name)
        image_labels = np.load(y_name)

    return image_list, image_labels


def get_model(train_all_layers=False):
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, kernel_initializer='uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dense(512, kernel_initializer='uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dense(3, kernel_initializer='uniform')(x)
    x = BatchNormalization()(x)
    predictions = Activation('softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    if not train_all_layers:
        for layer in base_model.layers:
            layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', metrics.categorical_accuracy])

    return model


def train_model(model,
                epochs=50,
                batch_size=64,
                train_steps_per_epoch=None,
                validation_steps=None):
    # Data augmentation generators
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       horizontal_flip=True,
                                       rotation_range=10)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Actual generators
    x_train, y_train = load_dataset(TRAIN_DATA, 'train')
    y_train = to_categorical(y_train, num_classes=3)
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

    x_validation, y_validation = load_dataset(VALIDATION_DATA, 'validation')
    y_validation = to_categorical(y_validation, num_classes=3)
    validation_generator = test_datagen.flow(x_validation, y_validation, batch_size=batch_size)

    callbacks = [
        TensorBoard(),
        EarlyStopping(patience=4),
        ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    ]

    if train_steps_per_epoch is None:
        train_steps_per_epoch = len(x_train) // batch_size

    if validation_steps is None:
        validation_steps = len(x_validation) // batch_size

    model.fit_generator(train_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        callbacks=callbacks)


def resume_training(weight_file_path,
                    epochs=50,
                    batch_size=64,
                    train_steps_per_epoch=2000,
                    validation_steps=800):
    model = load_model(weight_file_path)
    train_model(model, epochs, batch_size, train_steps_per_epoch, validation_steps)


if __name__ == '__main__':
    # m = get_model()
    # train_model(m, batch_size=16)
    # file = './weights.01-1.54.hdf5'
    # resume_training(file, batch_size=16, epochs=49)
    m = load_model('./weights.02-1.11.hdf5')
    x_test, y_test = load_dataset(TEST_DATA, 'test')
    print(m.predict(x_test))


