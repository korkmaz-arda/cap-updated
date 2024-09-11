import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


def hw_flatten(x):
    """Flatten the height and width dimensions into a single dimension."""
    x_shape = K.shape(x)
    return K.reshape(x, [x_shape[0], -1, x_shape[-1]])  # return [BATCH, W*H, CHANNELS]


class SelfAttention(Layer):
    def __init__(self, filters, **kwargs):
        self.filters = filters
        super(SelfAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Gamma is a trainable parameter initialized to 0.0
        self.gamma = self.add_weight(shape=(1,),
                                     initializer='zeros', 
                                     trainable=True,
                                     name='gamma')
        super(SelfAttention, self).build(input_shape)
        
    def call(self, inputs):
        """
        Expecting inputs to be a list: [img, f, g, h]
        - img: The original image tensor
        - f, g, h: Convolutionally reduced versions of img
        """
        img, f, g, h = inputs
        
        # Attention map
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # [batch_size, N, N]
        beta = tf.nn.softmax(s, axis=-1)  # Attention weights [batch_size, N, N]
        
        # Applying attention
        o = tf.matmul(beta, hw_flatten(h))  # [batch_size, N, C]
        o = K.reshape(o, shape=[K.shape(img)[0], K.shape(img)[1], K.shape(img)[2], self.filters])  # Reshape to original image shape
        
        # Output: weighted combination of input and attention-applied tensor
        img_out = self.gamma * o + img  # Apply attention modulation

        return img_out

    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def get_config(self):
        config = {'filters': self.filters}
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
