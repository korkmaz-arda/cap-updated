import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import glorot_normal, zeros
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.constraints import Constraint

class SeqSelfAttention(Layer):
    def __init__(self,
                 units=32,                 
                 return_attention=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attention_activation=None,
                 **kwargs):
        """Layer initialization.

        For additive attention, see: https://arxiv.org/pdf/1806.01264.pdf

        :param units: The dimension of the vectors used to calculate attention weights.
        :param return_attention: Whether to return the attention weights for visualization.
        :param kernel_initializer: The initializer for weight matrices.
        :param bias_initializer: The initializer for biases.
        :param kernel_regularizer: The regularization for weight matrices.
        :param bias_regularizer: The regularization for biases.
        :param kernel_constraint: The constraint for weight matrices.
        :param bias_constraint: The constraint for biases.
        :param attention_activation: The activation used for calculating attention weights.
        :param kwargs: Parameters for parent class.
        """
        super(SeqSelfAttention, self).__init__(**kwargs)
        self.units = units
        self.return_attention = return_attention
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.attention_activation = tf.keras.activations.get(attention_activation)

        self.Wx, self.Wt, self.bh = None, None, None
        self.Wa, self.ba = None, None

    def get_config(self):
        config = {
            'units': self.units,
            'return_attention': self.return_attention,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
            'attention_activation': tf.keras.activations.serialize(self.attention_activation),
        }
        base_config = super(SeqSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self._build_attention(input_shape)
        super(SeqSelfAttention, self).build(input_shape)

    def _build_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        
        self.bh = self.add_weight(shape=(self.units,),
                                  name='{}_Add_bh'.format(self.name),
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        
        self.ba = self.add_weight(shape=(1,),
                                  name='{}_Add_ba'.format(self.name),
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)

    def call(self, inputs, mask=None, **kwargs):
        alpha = self._emission(inputs)

        if self.attention_activation is not None:
            alpha = self.attention_activation(alpha)

        # Compute attention weights (softmax over alpha)
        alpha = K.exp(alpha - K.max(alpha, axis=-1, keepdims=True))
        a = alpha / K.sum(alpha, axis=-1, keepdims=True)

        # Compute the context vector (weighted sum of input vectors)
        c_r = K.batch_dot(a, inputs)

        if self.return_attention:
            return [c_r, a]
        return c_r

    def _emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        # Compute beta (attention scores)
        q = K.expand_dims(K.dot(inputs, self.Wt), 2)
        k = K.expand_dims(K.dot(inputs, self.Wx), 1)        
        beta = K.tanh(q + k + self.bh)

        # Compute alpha (attention logits)
        alpha = K.reshape(K.dot(beta, self.Wa) + self.ba, (batch_size, input_len, input_len))

        return alpha

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    @staticmethod
    def get_custom_objects():
        return {'SeqSelfAttention': SeqSelfAttention}