from keras.layers import Flatten, Input
from keras.models import Model
from layers import make_dense

def make_tiny(activation, input_shape, output_size, dropout=True, batch_norm=False, **kwargs):
    x = Input(shape=input_shape)
    y = x

    y = Flatten()(y)
    y = make_dense(output_size, activation=activation, **kwargs)(y)
    y = make_dense(output_size, activation='softmax', **kwargs)(y)

    return Model(inputs=x, outputs=y)
