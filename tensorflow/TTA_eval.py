import cv2, os, sys
import numpy as np
from utils import utils
from layers import squeeze_excitation_layer, mish
from model import InceptionV3_ConvNet as V3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score

HEIGHT = 250
WIDTH = 299
N_CLASSES = 5+1

def have_dataset(types='train', num_cls=N_CLASSES):
    if types=='train':
        X, y = np.load('X.npy'), np.load('y.npy')
        mX = np.load('mX.npy')
        y_train = utils.load_label(y, num_cls)
        X2, mX2 = flip.fliplr_image(X), flip.fliplr_image(mX)
        X_ = np.vstack([X, X2])
        mX_ = np.vstack([mX, mX2])
        y_train = np.vstack([y_train, y_train])
        print(X_.shape, mX_.shape, y_train.shape, X_.max(), X_.min())
        return X_, mX_, y_train
    elif types=='valid':
        print('valid')
        X_val, y_val = np.load('X_val.npy'), np.load('y_val.npy')
        mX_val = np.load('mX_val.npy')
        y_test = utils.load_label(y_val, num_cls)
        print(X_val.shape, mX_val.shape, y_test.shape, X_val.max(), X_val.min())
        return X_val, mX_val, y_test



def load_model(weight_path=None, use_se=True):
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    sgd = SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    input_shape = (HEIGHT, WIDTH, 3)
    n_classes = N_CLASSES
    inputs = Input(shape=input_shape)
    meta_input = Input(shape=input_shape)
     
    # layer1
    x = V3.InceptionV3(inputs)
    x = Dropout(rate=0.3)(x)
    # layer2
    flipx = V3.InceptionV3(meta_input)
    flipx = Dropout(rate=0.3)(flipx)
    
    x = Add()([x, flipx])
    if use_se:
        x = squeeze_excitation_layer.channel_spatial_squeeze_excite(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('sigmoid')(x)
    x = Dropout(rate=0.3)(x)
    o = Dense(n_classes, activation='softmax')(x)
    models = Model(inputs=[inputs, meta_input], outputs=o)
    models.summary()
    if weight_path:
        models.load_weights(os.path.join('weights', weight_path))
    models.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    models.summary()
    return models

def tta(model, X_test, y_test, bs):
    test_datagen = ImageDataGenerator(rotation_range=10,
    fill_mode='constant')

    tta_steps = 10
    predictions = []
    for i in tqdm(range(tta_steps)):
        preds = model.predict_generator(test_datagen.flow(X_test, batch_size=bs, shuffle=False), steps = len(y_test)/bs)
        predictions.append(preds)
    pred = np.mean(predictions, axis=0)
    f_pred=np.argmax(pred, axis=1)
    y_test = np.argmax(y_test, axis=-1)
    print('TTA Test acuracy:{}'.format(accuracy_score(y_test, f_pred)))
    #print('accracy', np.mean(np.equal(np.argmax(y_test, axis=-1), np.argmax(pred, axis=-1))))
    
def tta_evaluation():
    X, mX, y_train = have_dataset(types='train', num_cls=N_CLASSES)
    X_val, mX_val, y_test = have_dataset(types='valid', num_cls=N_CLASSES)
    path = "haircutep100_84.h5"
    models = load_model(weight_path=path, use_se=True)
    
    _, acc = models.evaluate([X_val, mX_val], y_test, verbose=0)
    print('\nTest accuracy: {0}'.format(acc))
    tta(model=models, X_test=[X_val, mX_val], y_test=y_test, bs=4)

    
if __name__=='__main__':
    tta_evaluation()
