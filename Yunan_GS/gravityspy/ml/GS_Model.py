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






