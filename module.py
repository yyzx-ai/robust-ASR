import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras


# ==============================================================================
# =                                  networks                                  =
# ==============================================================================

def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


def ConvGenerator(input_shape=(1, 1, 128),
                  output_channels=1,
                  dim=64,
                  n_upsamplings=3,
                  norm='batch_norm',
                  name='ConvGenerator'):
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)
    #print('\n\n----------------------h:%s'%h)
    #print('\noutput_size1:{}\n'.format(h.shape))
    # 1: 1x1 -> 4x4
    d = min(dim * 2 ** (n_upsamplings - 1), dim * 8)
    h = keras.layers.Conv2DTranspose(d, (13,4), strides=1, padding='valid', use_bias=False)(h)
    #print('\noutput_size2:{}\n'.format(h.shape))
    h = Norm()(h)
    #print('\noutput_size3:{}\n'.format(h.shape))
    h = tf.nn.tanh(h)  # or h = keras.layers.ReLU()(h)
    #print('\noutput_size4:{}\n'.format(h.shape))
    # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> ...
    for i in range(n_upsamplings - 1):
        d = min(dim * 2 ** (n_upsamplings - 2 - i), dim * 8)
        h = keras.layers.Conv2DTranspose(d, 4, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.tanh(h)  # or h = keras.layers.ReLU()(h)
    #print('\noutput_size5:{}\n'.format(h.shape))
    h = keras.layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same')(h)
    h = tf.nn.tanh(h)  # or h = keras.layers.Activation('tanh')(h)

    #print('\noutput_size6:{}\n'.format(h.shape))
    return keras.Model(inputs=inputs, outputs=h, name=name)


def ConvDiscriminator(input_shape=(104,32,1),
                      dim=64,
                      n_downsamplings=3,
                      norm='batch_norm',
                      name='ConvDiscriminator'):
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)
    #print('\noutput_size7:{}\n'.format(h.shape))
    # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
    h = keras.layers.Conv2D(dim, 4, strides=1, padding='same')(h)
    #h = keras.layers.Activation('tanh')(h)  # or keras.layers.LeakyReLU(alpha=0.2)(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)
    for i in range(n_downsamplings - 1):
        d = min(dim * 2 ** (i + 1), dim * 8)
        h = keras.layers.Conv2D(d, 4, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2) # or h = tf.nn.tanh(h) 

    # 2: logit
    h = keras.layers.Conv2D(1, 4, strides=2, padding='valid')(h)

    #print('\noutput_size8:{}\n'.format(h.shape))
    return keras.Model(inputs=inputs, outputs=h, name=name)
