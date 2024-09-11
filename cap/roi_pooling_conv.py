import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
        rois_mat: predefined ROIs matrix (coordinates) with shape (num_rois, 4)
    # Input shape
        4D tensor [X_img] with shape:
        X_img: `(1, rows, cols, channels)`
    # Output shape
        3D tensor with shape: `(1, num_rois, channels, pool_size, pool_size)`
    '''

    def __init__(self, pool_size, num_rois, rois_mat=None, **kwargs):
        self.pool_size = pool_size
        self.num_rois = num_rois
        self.rois_mat = rois_mat  # Predefined ROIs passed as a numpy array
        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        # Assuming TensorFlow dimension ordering
        self.nb_channels = input_shape[-1]  # The number of channels of the input image
        super(RoiPoolingConv, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, inputs):
        img = inputs  # Now, we only expect the image input (no ROIs input since we use rois_mat)

        input_shape = K.shape(img)
        outputs = []

        for roi_idx in range(self.num_rois):
            # Get the predefined ROI from the rois_mat
            x = self.rois_mat[roi_idx, 0]
            y = self.rois_mat[roi_idx, 1]
            w = self.rois_mat[roi_idx, 2]
            h = self.rois_mat[roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # Resize the cropped region to the desired pool size
            rs = tf.image.resize(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (-1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size, 'num_rois': self.num_rois, 'rois_mat': self.rois_mat}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Test Script: Modify for `rois_mat`
if __name__ == "__main__":
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    import numpy as np

    # Define inputs
    img_input = Input(shape=(224, 224, 3))  # Example input image shape

    # Predefined ROIs (rois_mat) for testing
    rois_mat = np.array([[0, 0, 50, 50], [50, 50, 100, 100], [100, 100, 150, 150], [150, 150, 200, 200]])

    # Create the RoiPoolingConv layer with predefined rois_mat
    roi_pooling_layer = RoiPoolingConv(pool_size=7, num_rois=4, rois_mat=rois_mat)(img_input)

    # Define the model
    model = Model(inputs=img_input, outputs=roi_pooling_layer)

    # Compile and summarize the model
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.summary()
