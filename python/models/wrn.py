from keras.layers import Add, BatchNormalization, Dropout, Flatten, GlobalMaxPooling2D, Input, MaxPooling2D
from keras.models import Model
from keras import backend as K
from layers import make_activation, make_conv, make_dense, make_dropout

# With BLU, this model achieves 96.08% accuracy on CIFAR-10.

def _make_block(filters, activation=None, batch_norm=True, dropout=0, strides=(1, 1), residual=True, channel_index=-1, **kwargs):
    def f(x):
        in_channels = K.int_shape(x)[channel_index]

        if in_channels == filters:
            y = x
            if batch_norm:
                y = BatchNormalization()(y)
            y = make_activation(activation, **kwargs)(y)
        else:
            if batch_norm:
                x = BatchNormalization()(x)
            x = make_activation(activation, **kwargs)(x)
            y = x

        y = make_conv(filters, activation=activation, strides=strides, **kwargs)(y)
        if batch_norm:
            y = BatchNormalization()(y)
        y = make_activation(activation, **kwargs)(y)
        if dropout != 0:
            y = make_dropout(dropout, activation=activation)(y)

        y = make_conv(filters, activation=activation, **kwargs)(y)

        if not residual:
            return y

        shortcut = x if in_channels == filters else make_conv(filters, shape=(1, 1), strides=strides, **kwargs)(x)
        return Add()([y, shortcut])
    return f

def make_wrn(activation, input_shape, output_size, depth=4, k=2, dropout=True, batch_norm=True, residual=True, **kwargs):
    assert((depth - 4) % 6 == 0)
    n = (depth - 4) / 6
    dropout = 0.3 if dropout else 0

    x = Input(shape=input_shape)
    y = x

    y = make_conv(16, activation=activation, **kwargs)(y)
    for _ in range(n):
        y = _make_block(16*k, activation=activation, batch_norm=batch_norm, dropout=dropout, residual=residual, **kwargs)(y)

    y = _make_block(32*k, activation=activation, batch_norm=batch_norm, dropout=dropout, strides=(2, 2), residual=residual, **kwargs)(y)
    for _ in range(n-1):
        y = _make_block(32*k, activation=activation, batch_norm=batch_norm, dropout=dropout, residual=residual, **kwargs)(y)

    y = _make_block(64*k, activation=activation, batch_norm=batch_norm, dropout=dropout, strides=(2, 2), residual=residual, **kwargs)(y)
    for _ in range(n-1):
        y = _make_block(64*k, activation=activation, batch_norm=batch_norm, dropout=dropout, residual=residual, **kwargs)(y)

    if batch_norm:
        y = BatchNormalization()(y)
    y = make_activation(activation, **kwargs)(y)
    y = GlobalMaxPooling2D()(y)
    y = make_dense(output_size, activation='softmax', **kwargs)(y)

    return Model(inputs=x, outputs=y)
