from keras.layers import BatchNormalization, Dropout, Flatten, GlobalMaxPooling2D, Input, MaxPooling2D
from keras.models import Model
from layers import make_conv, make_dense, make_dropout

# A ~5M parameter model inspired by FitNet-4 (Romero et. al).
# The primary difference is that the number of filters is doubled throughout,
# and the dense layer at the end is 2500 rather than 500.
# The change was inspired by the fact that this is approximately the topology used
# when Maxout is the activation, and I wanted a "fair" comparison for other activations.
# With BLU, this model achieves 94.4% accuracy on CIFAR-10.

def make_fitnet(activation, input_shape, output_size, dropout=True, batch_norm=False, **kwargs):
    x = Input(shape=input_shape)
    y = x

    if batch_norm:
        y = BatchNormalization()(y)
    if dropout:
        y = make_dropout(0.1, activation=activation)(y)

    y = make_conv(64, activation=activation, batch_norm=batch_norm, **kwargs)(y)
    y = make_conv(64, activation=activation, batch_norm=batch_norm, **kwargs)(y)
    y = make_conv(64, activation=activation, batch_norm=batch_norm, **kwargs)(y)
    y = make_conv(96, activation=activation, batch_norm=batch_norm, **kwargs)(y)
    y = make_conv(96, activation=activation, batch_norm=batch_norm, **kwargs)(y)
    y = MaxPooling2D()(y)
    if dropout:
        y = make_dropout(0.2, activation=activation)(y)

    y = make_conv(160, activation=activation, batch_norm=batch_norm, **kwargs)(y)
    y = make_conv(160, activation=activation, batch_norm=batch_norm, **kwargs)(y)
    y = make_conv(160, activation=activation, batch_norm=batch_norm, **kwargs)(y)
    y = make_conv(160, activation=activation, batch_norm=batch_norm, **kwargs)(y)
    y = make_conv(160, activation=activation, batch_norm=batch_norm, **kwargs)(y)
    y = MaxPooling2D()(y)
    if dropout:
        y = make_dropout(0.3, activation=activation)(y)

    y = make_conv(256, activation=activation, batch_norm=batch_norm, **kwargs)(y)
    y = make_conv(256, activation=activation, batch_norm=batch_norm, **kwargs)(y)
    y = make_conv(256, activation=activation, batch_norm=batch_norm, **kwargs)(y)
    y = make_conv(256, activation=activation, batch_norm=batch_norm, **kwargs)(y)
    y = make_conv(256, activation=activation, batch_norm=batch_norm, **kwargs)(y)
    y = GlobalMaxPooling2D()(y)
    if dropout:
        y = make_dropout(0.4, activation=activation)(y)

    y = make_dense(2500, activation=activation, batch_norm=batch_norm, **kwargs)(y)
    if dropout:
        y = make_dropout(0.5, activation=activation)(y)
    y = make_dense(output_size, activation='softmax', **kwargs)(y)

    return Model(inputs=x, outputs=y)
