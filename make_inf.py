import numpy as np
import cv2
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from PIL import Image, ImageOps


def model_inference(xrimage, model_weights):

    # Load the model 
    base_model = DenseNet121(weights = 'densenet.hdf5', include_top = False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(14, activation="sigmoid")(x)
    model = Model(inputs= base_model.input, outputs=predictions)

    model.load_weights(model_weights)

    image_raw = np.ndarray(shape = (1,320,320,3), dtype=np.float32)
    size = (320,320)

    image = xrimage
    image = ImageOps.fit(image,size, Image.ANTIALIAS)
    image_arr = np.asarray(image)
    image_arr = (image_arr.astype(np.float32) / 127.0) - 1
    image_raw[0] = np.expand_dims(image_arr, axis = 2)

    pred = model.predict(image_raw)

    return pred

def grad_cam(image, cls, layer_name = "bn", H=320, W=320):
    base_model = DenseNet121(weights = 'densenet.hdf5', include_top = False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(14, activation="sigmoid")(x)
    input_model = Model(inputs= base_model.input, outputs=predictions)

    input_model.load_weights('pretrained_model.h5')

    image_raw = np.ndarray(shape = (1,320,320,3), dtype=np.float32)
    size = (320,320)

    image = ImageOps.fit(image,size, Image.ANTIALIAS)
    image_arr = np.asarray(image)
    image_arr = (image_arr.astype(np.float32) / 127.0) - 1
    image_raw[0] = np.expand_dims(image_arr, axis = 2)


    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image_raw])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam

