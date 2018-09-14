from keras.layers import Add, AveragePooling2D, BatchNormalization, Concatenate, Conv2D, Cropping2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, ZeroPadding2D
from keras.models import Model
from keras import backend as K
from layers import make_activation, make_conv, make_dense, make_dropout

def _make_branch(filters, activation=None, strides=(1, 1), batch_norm=True, **kwargs):
    def f(x):
        y = make_activation(activation, **kwargs)(x)
        y = make_conv(filters, strides=strides)(y)
        if batch_norm:
            y = BatchNormalization()(y)
        y = make_activation(activation, **kwargs)(y)
        y = make_conv(filters)(y)
        if batch_norm:
            y = BatchNormalization()(y)
        return y
    return f

def _make_shortcut(filters, activation=None, strides=(1, 1), batch_norm=True, channel_index=-1, **kwargs):
    def f(x):
        in_channels = K.int_shape(x)[channel_index]
        if in_channels == filters:
            y = x
        else:
            y = make_activation(activation, **kwargs)(x)

            y1 = AveragePooling2D((1, 1), strides=strides, padding='same')(y)
            y1 = make_conv(filters/2, shape=(1, 1))(y1)

            y2 = ZeroPadding2D(padding=((1, 0), (1, 0)))(y)
            y2 = Cropping2D(cropping=((0, 1), (0, 1)))(y2)
            y2 = AveragePooling2D((1, 1), strides=strides, padding='same')(y2)
            y2 = make_conv(filters/2, shape=(1, 1))(y2)

            y = Concatenate(axis=channel_index)([y1, y2])
            if batch_norm:
                y = BatchNormalization()(y)
        return y
    return f

def _make_block(filters, activation=None, strides=(1, 1), batch_norm=True, residual=True, channel_index=-1, batch_size=64, **kwargs):
    def f(x):
        train_alpha = K.random_uniform((batch_size,))
        test_alpha = K.ones((batch_size,)) * 0.5
        alpha = K.in_train_phase(train_alpha, test_alpha)
        alpha = K.expand_dims(K.expand_dims(K.expand_dims(alpha)))

        b1 = _make_branch(filters, activation=activation, strides=strides, batch_norm=batch_norm, **kwargs)(x)
        b1 = Lambda(lambda x: x * alpha)(b1)

        b2 = _make_branch(filters, activation=activation, strides=strides, batch_norm=batch_norm, **kwargs)(x)
        b2 = Lambda(lambda x: x * (1. - alpha))(b2)

        if not residual:
            return Add()([b1, b2])

        sc = _make_shortcut(filters, activation=activation, strides=strides, batch_norm=batch_norm, channel_index=channel_index, **kwargs)(x)
        return Add()([b1, b2, sc])
    return f

def make_shakeshake(activation, input_shape, output_size, depth=26, k=32, batch_norm=True, residual=True, **kwargs):
    assert((depth - 2) % 6 == 0)
    n = (depth - 2) / 6

    x = Input(batch_shape=(kwargs['batch_size'],) + input_shape)
    y = make_conv(16)(x)
    y = BatchNormalization()(y)

    for _ in range(n):
        y = _make_block(k, activation=activation, batch_norm=batch_norm, residual=residual, **kwargs)(y)

    y = _make_block(2*k, activation=activation, batch_norm=batch_norm, residual=residual, strides=(2, 2), **kwargs)(y)
    for _ in range(n-1):
       y = _make_block(2*k, activation=activation, batch_norm=batch_norm, residual=residual, **kwargs)(y)

    y = _make_block(4*k, activation=activation, batch_norm=batch_norm, residual=residual, strides=(2, 2), **kwargs)(y)
    for _ in range(n-1):
        y = _make_block(4*k, activation=activation, batch_norm=batch_norm, residual=residual, **kwargs)(y)

    y = make_activation(activation, **kwargs)(y)
    y = GlobalAveragePooling2D()(y)
    y = make_dense(output_size, activation='softmax', **kwargs)(y)

    return Model(inputs=x, outputs=y)
