import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Multiply, Permute
from tensorflow.keras import backend as K

def squeeze_excite_block(input_tensor, ratio=16):
    ''' Create a squeeze-excite block

    Args:
        input_tensor: input tensor
        ratio: reduction ratio for squeeze operation

    Returns:
        A TensorFlow/Keras tensor
    '''

    # Determine the channel axis
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    # Get the number of filters (channels)
    filters = input_tensor.shape[channel_axis]

    # Squeeze operation: Global average pooling
    se = GlobalAveragePooling2D()(input_tensor)
    
    # Reshape the tensor to match the input shape: (1, 1, channels)
    se = Reshape((1, 1, filters))(se)

    # Excitation operation: Two dense layers
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    # Adjust for channels_first format if necessary
    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    # Scale the input_tensor by the se tensor
    x = Multiply()([input_tensor, se])

    return x