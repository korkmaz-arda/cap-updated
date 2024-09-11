import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, Conv3D, Conv2DTranspose, Embedding, InputSpec
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers, constraints


class DenseSN(Dense):   
    def build(self, input_shape):
        # Call the parent class build method (this automatically handles kernel and bias creation)
        super(DenseSN, self).build(input_shape)

        # Spectral normalization u vector (based on the shape of the kernel from the parent build)
        self.u = self.add_weight(shape=(1, self.kernel.shape[-1]),
                                initializer=tf.keras.initializers.RandomNormal(0, 1),
                                name='sn_u',
                                trainable=False)

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

        def power_iteration(W, u):
            v = _l2normalize(tf.linalg.matmul(u, tf.transpose(W)))
            u = _l2normalize(tf.linalg.matmul(v, W))
            return u, v

        W_shape = tf.shape(self.kernel)
        W_reshaped = tf.reshape(self.kernel, [-1, W_shape[-1]])
        u, v = power_iteration(W_reshaped, self.u)
        sigma = tf.linalg.matmul(v, W_reshaped)
        sigma = tf.linalg.matmul(sigma, tf.transpose(u))
        W_bar = W_reshaped / sigma
        W_bar = tf.reshape(W_bar, W_shape)

        # Update spectral normalization vector
        if training:
            self.u.assign(u)

        output = tf.matmul(inputs, W_bar)
        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            return self.activation(output)
        return output


class ConvSN2D(Conv2D):
    def build(self, input_shape):
        super(ConvSN2D, self).build(input_shape)
        self.u = self.add_weight(shape=(1, self.kernel.shape[-1]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn_u',
                                 trainable=False)

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)

        def power_iteration(W, u):
            v = _l2normalize(tf.linalg.matmul(u, tf.transpose(W)))
            u = _l2normalize(tf.linalg.matmul(v, W))
            return u, v

        W_shape = tf.shape(self.kernel)
        W_reshaped = tf.reshape(self.kernel, [-1, W_shape[-1]])
        u, v = power_iteration(W_reshaped, self.u)
        sigma = tf.linalg.matmul(v, W_reshaped)
        sigma = tf.linalg.matmul(sigma, tf.transpose(u))
        W_bar = W_reshaped / sigma
        W_bar = tf.reshape(W_bar, W_shape)

        if training:
            self.u.assign(u)

        outputs = K.conv2d(inputs,
                           W_bar,
                           strides=self.strides,
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class ConvSN1D(Conv1D):
    def build(self, input_shape):
        super(ConvSN1D, self).build(input_shape)
        self.u = self.add_weight(shape=(1, self.kernel.shape[-1]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn_u',
                                 trainable=False)

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)

        def power_iteration(W, u):
            v = _l2normalize(tf.linalg.matmul(u, tf.transpose(W)))
            u = _l2normalize(tf.linalg.matmul(v, W))
            return u, v

        W_shape = tf.shape(self.kernel)
        W_reshaped = tf.reshape(self.kernel, [-1, W_shape[-1]])
        u, v = power_iteration(W_reshaped, self.u)
        sigma = tf.linalg.matmul(v, W_reshaped)
        sigma = tf.linalg.matmul(sigma, tf.transpose(u))
        W_bar = W_reshaped / sigma
        W_bar = tf.reshape(W_bar, W_shape)

        if training:
            self.u.assign(u)

        outputs = K.conv1d(inputs,
                           W_bar,
                           strides=self.strides,
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class ConvSN3D(Conv3D):
    def build(self, input_shape):
        super(ConvSN3D, self).build(input_shape)
        self.u = self.add_weight(shape=(1, self.kernel.shape[-1]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn_u',
                                 trainable=False)

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)

        def power_iteration(W, u):
            v = _l2normalize(tf.linalg.matmul(u, tf.transpose(W)))
            u = _l2normalize(tf.linalg.matmul(v, W))
            return u, v

        W_shape = tf.shape(self.kernel)
        W_reshaped = tf.reshape(self.kernel, [-1, W_shape[-1]])
        u, v = power_iteration(W_reshaped, self.u)
        sigma = tf.linalg.matmul(v, W_reshaped)
        sigma = tf.linalg.matmul(sigma, tf.transpose(u))
        W_bar = W_reshaped / sigma
        W_bar = tf.reshape(W_bar, W_shape)

        if training:
            self.u.assign(u)

        outputs = K.conv3d(inputs,
                           W_bar,
                           strides=self.strides,
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class ConvSN2DTranspose(Conv2DTranspose):
    def build(self, input_shape):
        super(ConvSN2DTranspose, self).build(input_shape)
        self.u = self.add_weight(shape=(1, self.kernel.shape[-1]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn_u',
                                 trainable=False)

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)

        def power_iteration(W, u):
            v = _l2normalize(tf.linalg.matmul(u, tf.transpose(W)))
            u = _l2normalize(tf.linalg.matmul(v, W))
            return u, v

        W_shape = tf.shape(self.kernel)
        W_reshaped = tf.reshape(self.kernel, [-1, W_shape[-1]])
        u, v = power_iteration(W_reshaped, self.u)
        sigma = tf.linalg.matmul(v, W_reshaped)
        sigma = tf.linalg.matmul(sigma, tf.transpose(u))
        W_bar = W_reshaped / sigma
        W_bar = tf.reshape(W_bar, W_shape)

        if training:
            self.u.assign(u)

        outputs = K.conv2d_transpose(
            inputs,
            W_bar,
            output_shape=self.compute_output_shape(K.shape(inputs)),
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class EmbeddingSN(Embedding):
    def build(self, input_shape):
        super(EmbeddingSN, self).build(input_shape)
        self.u = self.add_weight(shape=(1, self.embeddings.shape[-1]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn_u',
                                 trainable=False)

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)

        def power_iteration(W, u):
            v = _l2normalize(tf.linalg.matmul(u, tf.transpose(W)))
            u = _l2normalize(tf.linalg.matmul(v, W))
            return u, v

        W_shape = tf.shape(self.embeddings)
        W_reshaped = tf.reshape(self.embeddings, [-1, W_shape[-1]])
        u, v = power_iteration(W_reshaped, self.u)
        sigma = tf.linalg.matmul(v, W_reshaped)
        sigma = tf.linalg.matmul(sigma, tf.transpose(u))
        W_bar = W_reshaped / sigma
        W_bar = tf.reshape(W_bar, W_shape)

        if training:
            self.u.assign(u)

        return tf.gather(W_bar, inputs)
