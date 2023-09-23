import argparse
import cv2, os, sys, glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from utils import utils

from random_eraser.random_eraser import get_random_eraser
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
HEIGHT = 250
WIDTH = 299
N_CLASSES = 5+1

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='batch size', default=4, type=int)
    parser.add_argument('--num_ep', help='num epochs', default=101, type=int)
    parser.add_argument('--weight_dir', help='folder path to save trained weight', default='weights', type=str)
    return parser

def gene_flow_inputs(datagen, x_image, y_train, batch=2):
    batch = datagen.flow(x_image, y_train, batch_size=batch)
    while True:
        batch_image, batch_mask = batch.next()
        yield batch_image, batch_mask
        
        
def have_dataset(types='train', num_cls=N_CLASSES, batch_size=4):
    generators = ImageDataGenerator(rotation_range=10,
    fill_mode='constant', preprocessing_function=get_random_eraser(v_l=0, v_h=1)))
    
    if types=='train':
        X, mX, y = np.load('X.npy'), np.load('mX.npy'), np.load('y.npy')
        y_train = utils.load_label(y, num_cls)
        training = gene_flow_inputs(generators, [X, mX], y_train, batch=batch_size)
        print(X.shape, mX.shape, y_train.shape, X.max(), X.min())
        return training, X
    elif types=='valid':
        print('valid')
        X_val, mX_val, y_val = np.load('X_val.npy'), np.load('mX_val.npy'), np.load('y_val.npy')
        y_test = utils.load_label(y_val, num_cls)
        # valid_generator = gene_flow_inputs(generators, [X_val, mX_val], y_test, batch=batch_size)
        print(X_val.shape, mX_val.shape, y_test.shape, X_val.max(), X_val.min())
        return X_val, mX_val, y_val



def load_model(weight_path=None, use_se=True):
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    sgd = SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    input_shape = (HEIGHT, WIDTH, 3)
    n_classes = N_CLASSES
    input_shape = (HEIGHT, WIDTH, 3)
    # layer1
    base_model = EfficientNetB0(input_shape=input_shape, weights='imagenet', include_top=True)
    x = base_model.get_layer('top_conv').output
    x = Dropout(rate=0.3)(x)
    x = GlobalAveragePooling2D()(x)
    x = Lambda(lambda x: mish(x))(x)
    x = Dropout(rate=0.2)(x)
    o = Dense(n_classes, activation='softmax')(x)
    models = Model(inputs=base_model.input, outputs=o)
    models.summary()
    if weight_path:
        models.load_weights(os.path.join('weights', weight_path))
    models.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    models.summary()
    return models

def train():
    opts = get_argparser().parse_args()
    BATCH_SIZE=opts.batch_size
    EPOCHS=opts.num_ep
    WEIGHT_DIR = opts.weight_dir
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    
    train_generator, X = have_dataset(types='train', num_cls=N_CLASSES, batch_size=BATCH_SIZE)
    X_val, mX_val, y_val = have_dataset(types='valid', num_cls=N_CLASSES, batch_size=BATCH_SIZE)
 
    models = load_model(weight_path=None, use_se=True)
    checkpoint_path = "weights/cp-haircut_{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, period=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=1)
    callback = [cp_callback, reduce_lr]

    print('train')
    startTime1 = datetime.now() #DB

    hist1 = models.fit(train_generator, steps_per_epoch=int(len(X)/BATCH_SIZE)*2,
      epochs=EPOCHS, validation_data=([X_val, mX_val], y_val), batch_size=BATCH_SIZE, verbose=1, callbacks=callback)

    endTime1 = datetime.now()
    diff1 = endTime1 - startTime1
    print("\n")
    print("Elapsed time for Keras training (s): ", diff1.total_seconds())
    print("\n")

    for key in ["loss", "val_loss"]:
        plt.plot(hist1.history[key],label=key)
    plt.legend()

    plt.savefig(os.path.join(WEIGHT_DIR, "model" + "_training_curves_" + str(WIDTH) + "x" + str(HEIGHT) + ".png"))

    models.save(os.path.join(WEIGHT_DIR, "ep" + str(EPOCHS) + "_trained_unet_model" + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".hdf5"))
    print("\nEnd of UNET training\n")
    score = models.evaluate([X_val, mX_val], y_val)
    print(score)
if __name__=='__main__':
    train()
