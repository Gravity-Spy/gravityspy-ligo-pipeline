import tensorflow as tf
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Concatenate


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

class Attention(Layer):
    
    def __init__(self, return_sequences=False, return_attention=False):
        self.return_sequences = return_sequences
        self.return_attention = return_attention
        super(Attention,self).__init__()
        
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(Attention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        if self.return_attention:
            return K.sum(output, axis=1), a
        
        return K.sum(output, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'return_sequences': self.return_sequences,
            'return_attention': self.return_attention 
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

  return x


def GS_new(nb_classes, img_rows, img_cols, channel_num, return_attention = False):

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

  att, weights = Attention(return_sequences=False, return_attention = True)(concat)

  output = tf.keras.layers.Dense(nb_classes, activation = 'softmax', name = 'Last_layer')(att)  

  if return_attention:
    final_model = tf.keras.models.Model(inputs=[input_tensor_0_5,input_tensor_1_0,input_tensor_2_0,input_tensor_4_0], outputs=[output, weights])

  else:
    final_model = tf.keras.models.Model(inputs=[input_tensor_0_5,input_tensor_1_0,input_tensor_2_0,input_tensor_4_0], outputs=output)
  return final_model

if __name__ == '__main__':
    model = GS_new(23, 600, 600, 3)
    print(model.summary())






