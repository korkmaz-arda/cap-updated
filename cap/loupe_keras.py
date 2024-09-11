"""
This code is modified from the following paper.
Learnable mOdUle for Pooling fEatures (LOUPE)
Contains a collection of models (NetVLAD, NetRVLAD, NetFV and Soft-DBoW)
which enables pooling of a list of features into a single compact 
representation.

Reference:

Learnable pooling method with Context Gating for video classification
Antoine Miech, Ivan Laptev, Josef Sivic

"""

import math
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

# Keras version

class NetRVLAD(layers.Layer):
    """Creates a NetRVLAD class (Residual-less NetVLAD)."""
    
    def __init__(self, feature_size, max_samples, cluster_size, output_dim, **kwargs):
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.cluster_size = cluster_size
        super(NetRVLAD, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.cluster_weights = self.add_weight(
            name='kernel_W1',
            shape=(self.feature_size, self.cluster_size),
            initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),
            trainable=True
        )
        
        self.cluster_biases = self.add_weight(
            name='kernel_B1',
            shape=(self.cluster_size,),
            initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),
            trainable=True
        )
        
        self.Wn = self.add_weight(
            name='kernel_H1',
            shape=(self.cluster_size * self.feature_size, self.output_dim),
            initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.cluster_size)),
            trainable=True
        )
        
        super(NetRVLAD, self).build(input_shape)  # Be sure to call this at the end
    
    def call(self, reshaped_input):
        """
        Forward pass of a NetRVLAD block.

        Args:
        reshaped_input: Should be reshaped in the following form:
        'batch_size' x 'max_samples' x 'feature_size'
        It should be reshaped in the following form:
        'batch_size*max_samples' x 'feature_size'

        Returns:
        vlad: the pooled vector of size: 'batch_size' x 'output_dim'
        """
        
        # Compute N_v in Equation 3 of the paper
        activation = K.dot(reshaped_input, self.cluster_weights)
        activation += self.cluster_biases
        activation = tf.nn.softmax(activation)

        # Reshape and transpose activation for VLAD computation
        activation = tf.reshape(activation, [-1, self.max_samples, self.cluster_size])
        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(reshaped_input, [-1, self.max_samples, self.feature_size])

        # Compute VLAD vector
        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.nn.l2_normalize(vlad, 1)
        vlad = tf.reshape(vlad, [-1, self.cluster_size * self.feature_size])
        Nv = tf.nn.l2_normalize(vlad, 1)

        # Equation 3 in the paper: \hat{y} = W_N N_v
        vlad = K.dot(Nv, self.Wn)

        return vlad

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)



if __name__ == "__main__":
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    # Example usage of NetRVLAD
    model = Sequential([
        # Assume we have an input with 100 samples, each with 128 features
        NetRVLAD(feature_size=128, max_samples=100, cluster_size=8, output_dim=64, input_shape=(100, 128)),
        Dense(10, activation='softmax')  # Example final layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Summary
    model.summary()
