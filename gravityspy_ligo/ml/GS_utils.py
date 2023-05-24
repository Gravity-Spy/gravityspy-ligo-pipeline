""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import MaxPooling2D, Conv2D

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Concatenate


#4/2/2018
def concatenate_views(image_set1, image_set2, image_set3,
                      image_set4, image_size, rgb_flag,
                      order_of_channels):
    """Create a merged view from a set of 4 views of one sample image

    Parameters:
        image_set1 (array):
            The grayscale downsamples pixels for one duration of one
            of the samples

        image_set2 (array):
            The grayscale downsamples pixels for one duration of one
            of the samples

        image_set3 (array):
            The grayscale downsamples pixels for one duration of one
            of the samples

        image_set4 (array):
            The grayscale downsamples pixels for one duration of one
            of the samples

        image_size (list):
            This refers to the shape of the non flattened pixelized image
            array

        rgb_flag (bool):
            Are you trying to create a merged view of grayscale
            or RGB image renders

    Returns:
        concated_view (array):
            A single merged view of the sample, in the case
            a merged view of the 0.5 1.0 2.0 and 4.0 duration
            omega scans.
    """
    img_rows = image_size[0]
    img_cols = image_size[1]
    if rgb_flag:
        ch = 3
    else:
        ch = 1

    if order_of_channels == 'channels_last':
        concat_images_set = np.zeros((len(image_set1), img_rows * 2, img_cols, ch))
        concat_images_set2 = np.zeros((len(image_set3), img_rows * 2, img_cols, ch))
        concat_row_axis = 0
        concat_col_axis = 2
    elif order_of_channels == 'channels_first':
        concat_images_set = np.zeros((len(image_set1), ch, img_rows * 2, img_cols))
        concat_images_set2 = np.zeros((len(image_set3), ch, img_rows * 2, img_cols))
        concat_row_axis = 1
        concat_col_axis = 3
    else:
        raise ValueError("Do not understand supplied channel order")

    assert len(image_set1) == len(image_set2)
    for i in range(0, len(image_set1)):
        concat_images_set[i, :, :, :] = np.append(image_set1[i, :, :, :], image_set2[i, :, :, :], axis=concat_row_axis)

    assert len(image_set3) == len(image_set4)
    for i in range(0, len(image_set3)):
        concat_images_set2[i, :, :, :] = np.append(image_set3[i, :, :, :], image_set4[i, :, :, :], axis=concat_row_axis)

    out = np.append(concat_images_set,concat_images_set2, axis=concat_col_axis)
    return out

#4/2/2018
def build_cnn(img_rows, img_cols, order_of_channels):
    """This is where we use Keras to build a covolutional neural network (CNN)

    The CNN built here is described in the
    `Table 5 <https://www.sciencedirect.com/science/article/pii/S0020025518301634#tbl0004>`_

    There are 5 layers. For each layer the logic is as follows

    input 2D matrix --> 2D Conv layer with x number of kernels with a 5 by 5 shape
    --> activation layer we use `ReLU <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_
    --> MaxPooling of size 2 by 2 all pixels are now grouped into bigger pixels of
    size 2 by 2 and the max pixel of the pixels that make up the 2 by 2 pizels is the
    value of the bigger pixel --> Dropout set to 50 percent. This means each pixel
    at this stage has a 50 percent chance of being set to 0.

    Parameters:
        image_rows (int):
            This refers to the number of rows in the non-flattened image

        image_cols (int):
            This refers to the number of cols in the non-flattened image

    Returns:
        model (`object`):
            a CNN
    """
    W_reg = 1e-4
    print('regularization parameter: ', W_reg)
    if order_of_channels == 'channels_last':
        input_shape = (img_rows, img_cols, 1)
    elif order_of_channels == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        raise ValueError("Do not understand supplied channel order")
    model = Sequential()
    model.add(Conv2D(16, (5, 5), padding='valid',
              input_shape=input_shape,
              kernel_regularizer=l2(W_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(32, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=l2(W_reg)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    print (model.summary())
    return model


def cosine_distance(vects):
    """Calculate the cosine distance of an array

    Parameters:

        vect (array):
    """
    x, y = vects
    x = K.maximum(x, K.epsilon())
    y = K.maximum(y, K.epsilon())
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return 1.0 - K.sum(x * y, axis=1, keepdims=True)

def siamese_acc(thred):
    """Calculate simaese accuracy

    Parameters:
        thred (float):
            It is something
    """
    def inner_siamese_acc(y_true, y_pred):
        pred_res = y_pred < thred
        acc = K.mean(K.cast(K.equal(K.cast(pred_res, dtype='int32'), K.cast(y_true, dtype='int32')), dtype='float32'))
        return acc

    return inner_siamese_acc

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def create_pairs3_gen(data, class_indices, batch_size):
    """ Create the pairs

    Parameters:
        data (float):
            It is something
        class_indices (list):
            It is something
        batch_size (int):
            It is something
    """
    pairs1 = []
    pairs2 = []
    labels = []
    number_of_classes = len(class_indices)
    counter = 0
    while True:
        for d in range(len(class_indices)):
            for i in range(len(class_indices[d])):
                counter += 1
                # positive pair
                j = random.randrange(0, len(class_indices[d]))
                z1, z2 = class_indices[d][i], class_indices[d][j]

                pairs1.append(data[z1])
                pairs2.append(data[z2])
                labels.append(1)

                # negative pair
                inc = random.randrange(1, number_of_classes)
                other_class_id = (d + inc) % number_of_classes
                j = random.randrange(0, len(class_indices[other_class_id])-1)
                z1, z2 = class_indices[d][i], class_indices[other_class_id][j]

                pairs1.append(data[z1])
                pairs2.append(data[z2])
                labels.append(0)

                if counter == batch_size:
                    #yield np.array(pairs), np.array(labels)
                    yield [np.asarray(pairs1, np.float32), np.asarray(pairs2, np.float32)], np.asarray(labels, np.int32)
                    counter = 0
                    pairs1 = []
                    pairs2 = []
                    labels = []


def split_data_set(data, fraction_validation=.125, fraction_testing=None,
                   image_size=[140, 170]):
    """Split data set to training validation and optional testing

    Parameters:
        data (str):
            Pickle file containing training set data

        fraction_validation (float, optional):
            Default .125

        fraction_testing (float, optional):
            Default None

        image_size (list, optional):
            Default [140, 170]

    Returns:
        numpy arrays
    """

    img_rows, img_cols = image_size[0], image_size[1]
    validationDF = data.groupby('Label').apply(
                       lambda x: x.sample(frac=fraction_validation,
                       random_state=random_seed)
                       ).reset_index(drop=True)

    data = data.loc[~data.uniqueID.isin(
                                    validationDF.uniqueID)]

    if fraction_testing:
        testingDF = data.groupby('Label').apply(
                   lambda x: x.sample(frac=fraction_testing,
                             random_state=random_seed)
                   ).reset_index(drop=True)

        data = data.loc[~data.uniqueID.isin(
                                        testingDF.uniqueID)]


    # concatenate the pixels
    train_set_x_1 = np.vstack(data['0.5.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
    validation_x_1 = np.vstack(validationDF['0.5.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)

    train_set_x_2 = np.vstack(data['1.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
    validation_x_2 = np.vstack(validationDF['1.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)

    train_set_x_3 = np.vstack(data['2.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
    validation_x_3 = np.vstack(validationDF['2.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)

    train_set_x_4 = np.vstack(data['4.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
    validation_x_4 = np.vstack(validationDF['4.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)

    if fraction_testing:
        testing_x_1 = np.vstack(testingDF['0.5.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
        testing_x_2 = np.vstack(testingDF['1.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
        testing_x_3 = np.vstack(testingDF['2.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
        testing_x_4 = np.vstack(testingDF['4.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)

    concat_train = concatenate_views(train_set_x_1, train_set_x_2,
                            train_set_x_3, train_set_x_4, [img_rows, img_cols], False)
    concat_valid = concatenate_views(validation_x_1, validation_x_2,
                            validation_x_3, validation_x_4,
                            [img_rows, img_cols], False)

    if fraction_testing:
        concat_test = concatenate_views(testing_x_1, testing_x_2,
                            testing_x_3, testing_x_4,
                            [img_rows, img_cols], False)
    else:
        concat_test = None

    return concat_train, concat_valid, concat_test

def inception_resnet_module(x, num_filters):

    # Inception module
    tower_1 = Conv2D(num_filters, (1,1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(num_filters, (1,1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(num_filters, (3,3), padding='same', activation='relu')(tower_2)
    tower_3 = Conv2D(num_filters, (1,1), padding='same', activation='relu')(x)
    tower_3 = Conv2D(num_filters, (5,5), padding='same', activation='relu')(tower_3)
    tower_4 = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_4 = Conv2D(num_filters, (1,1), padding='same', activation='relu')(tower_4)
    output = Concatenate()([tower_1, tower_2, tower_3, tower_4])

    # Residual connection
    shortcut = Conv2D(num_filters*4, (1,1), padding='same')(x)
    output = tf.keras.layers.add([shortcut, output])

    # Activation
    output = Activation('relu')(output)

    return output

class AttentionLayer(Layer):
    """Implementation of the attention-based Deep MIL layer.

    Args:
      weight_params_dim: Positive Integer. Dimension of the weight matrix.
      kernel_initializer: Initializer for the `kernel` matrix.
      kernel_regularizer: Regularizer function applied to the `kernel` matrix.
      use_gated: Boolean, whether or not to use the gated mechanism.

    Returns:
      List of 2D tensors with BAG_SIZE length.
      The tensors are the attention scores after softmax with shape `(batch_size, 1)`.
    """

    def __init__(
        self,
        weight_params_dim,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        use_gated=False,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

    def build(self, input_shape):

        # Input shape.
        # List of 2D tensors with shape: (batch_size, input_dim).
        input_dim = input_shape[-1]

        self.v_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.v_init,
            name="v",
            regularizer=self.v_regularizer,
            trainable=True,
        )

        self.w_weight_params = self.add_weight(
            shape=(self.weight_params_dim, 1),
            initializer=self.w_init,
            name="w",
            regularizer=self.w_regularizer,
            trainable=True,
        )

        if self.use_gated:
            self.u_weight_params = self.add_weight(
                shape=(input_dim, self.weight_params_dim),
                initializer=self.u_init,
                name="u",
                regularizer=self.u_regularizer,
                trainable=True,
            )
        else:
            self.u_weight_params = None

        self.input_built = True

    def call(self, inputs):

        # Assigning variables from the number of inputs.
        instances = self.compute_attention_scores(inputs)
        # instances = [self.compute_attention_scores(instance) for instance in inputs]

        # Apply softmax over instances such that the output summation is equal to 1.
        alpha = tf.math.softmax(instances, axis=1)

        return alpha

    def compute_attention_scores(self, instance):

        # Reserve in-case "gated mechanism" used.
        original_instance = instance

        # tanh(v*h_k^T)
        instance = tf.math.tanh(tf.tensordot(instance, self.v_weight_params, axes=1))

        # for learning non-linear relations efficiently.
        if self.use_gated:

            instance = instance * tf.math.sigmoid(
                tf.tensordot(original_instance, self.u_weight_params, axes=1)
            )

        # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))
        return tf.tensordot(instance, self.w_weight_params, axes=1)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'weight_params_dim': self.weight_params_dim,
            'use_gated': self.use_gated,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
        })
        return config

def build_baseline(x):

  x = Conv2D(16, (5, 5), padding='valid')(x)
  x = Activation("relu")(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size=(3, 3))(x)

  x = Conv2D(32, (5, 5), padding='valid')(x)
  x = Activation("relu")(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)

  x = inception_resnet_module(x,16)

  x = Conv2D(64, (5, 5), padding='valid')(x)
  x = Activation("relu")(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)

  x = inception_resnet_module(x,16)

  x = Conv2D(64, (5, 5), padding='valid')(x)
  x = Activation("relu")(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)

  x = inception_resnet_module(x,32)   

  x = Conv2D(128, (5, 5), padding='valid')(x)
  x = Activation("relu")(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)

  x = Dropout(0.25)(x)
  x = Flatten()(x)
  x = Dense(128)(x)
  x = Activation('relu')(x)
  x = tf.expand_dims(x, axis=1)
  # x = Dropout(0.25)(x)

  return x

def GS_new(nb_classes, img_rows, img_cols, channel_num):
    initializer = glorot_normal()

    input_tensor_0_5 = tf.keras.layers.Input(shape=(img_rows, img_cols, channel_num), name="0.5")
    input_tensor_1_0 = tf.keras.layers.Input(shape=(img_rows, img_cols, channel_num), name="1.0")
    input_tensor_2_0 = tf.keras.layers.Input(shape=(img_rows, img_cols, channel_num), name="2.0")
    input_tensor_4_0 = tf.keras.layers.Input(shape=(img_rows, img_cols, channel_num), name="4.0")
    
    x_0_5 = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(input_tensor_0_5)
    cnn_0_5 = build_baseline(x_0_5)

    x_1_0 = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(input_tensor_1_0)
    cnn_1_0 = build_baseline(x_1_0)

    x_2_0 = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(input_tensor_2_0)
    cnn_2_0 = build_baseline(x_2_0)

    x_4_0 = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(input_tensor_4_0)
    cnn_4_0 = build_baseline(x_4_0)

    concat = tf.concat([cnn_0_5, cnn_1_0, cnn_2_0, cnn_4_0], axis=1) 

    att = AttentionLayer(
        weight_params_dim=64,
        kernel_regularizer= tf.keras.regularizers.l2(0.01),
        use_gated=True,
        name="alpha",
    )(concat)

    intermediate = tf.linalg.matmul(att,concat, transpose_a = True) 
    intermediate = tf.squeeze(intermediate, axis=1)  
    output = tf.keras.layers.Dense(nb_classes, activation = 'softmax', name = 'Last_layer')(intermediate) 

    final_model = tf.keras.models.Model(inputs=[input_tensor_0_5,input_tensor_1_0,input_tensor_2_0,input_tensor_4_0], outputs=output)

    return final_model

if __name__ == '__main__':
    model = GS_new(23, 600, 600, 3)
    print(model.summary())
