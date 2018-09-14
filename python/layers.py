from activations import APLU, BLU, SoftExp, ScaledReLU
from keras.layers import Activation, Add, AlphaDropout, BatchNormalization, Conv2D, Dense, Dropout, Maximum
from keras.layers import LeakyReLU, PReLU
from keras import backend as K
from keras.regularizers import l2
from keras_contrib.layers.advanced_activations import PELU

def make_activation(activation, alpha=None, beta=None, share_params=False, blu_reg=0, aplu_segments=3, **kwargs):
    shared_axes = (1, 2) if share_params else None
    if activation == 'aplu':
        return APLU(segments=aplu_segments, shared_axes=shared_axes)
    elif activation == 'blu' or activation == 'blu-beta':
        return BLU(alpha=alpha or 0.5, shared_axes=shared_axes, beta_regularizer=l2(blu_reg))
    elif activation == 'blu-alpha':
        return BLU(beta=beta or 0.5, parametric_alpha=True, parametric_beta=False, shared_axes=shared_axes, alpha_regularizer=l2(blu_reg))
    elif activation == 'blu-alpha-beta':
        return BLU(parametric_alpha=True, shared_axes=shared_axes, alpha_regularizer=l2(blu_reg), beta_regularizer=l2(blu_reg))
    elif activation == 'blu-const':
        return BLU(alpha=alpha or 0.5, beta=beta or 0.5, parametric_beta=False, shared_axes=shared_axes)
    elif activation == 'lrelu':
        return LeakyReLU(alpha=alpha)
    elif activation == 'maxout':
        return Maximum()
    elif activation == 'pelu':
        return PELU(shared_axes=shared_axes)
    elif activation == 'prelu':
        return PReLU(shared_axes=shared_axes)
    elif activation == 'srelu':
        return ScaledReLU(alpha=alpha or 0.5, beta=beta or 1.5, shared_axes=shared_axes)
    elif activation == 'srelu-alpha':
        return ScaledReLU(parametric_alpha=True, beta=beta or 1.5, shared_axes=shared_axes)
    elif activation == 'srelu-beta':
        return ScaledReLU(alpha=alpha or 0.5, parametric_beta=True, shared_axes=shared_axes)
    elif activation == 'srelu-alpha-beta':
        return ScaledReLU(parametric_alpha=True, parametric_beta=True, shared_axes=shared_axes)
    elif activation == 'softexp':
        return SoftExp(shared_axes=shared_axes)
    else:
        return Activation(activation)

def make_conv(filters, shape=(3, 3), activation=None, pieces=2, batch_norm=False, alpha=None, beta=None, share_params=False, blu_reg=0, aplu_segments=3, **kwargs):
    if activation != 'maxout':
        pieces = 1

    def f(x):
        y = [Conv2D(filters, shape, padding='same', **kwargs)(x) for _ in range(pieces)]
        if batch_norm:
            y = [BatchNormalization()(p) for p in y]
        y = y[0] if len(y) == 1 else y
        if activation:
            y = make_activation(activation, alpha=alpha, beta=beta, share_params=share_params, blu_reg=blu_reg, aplu_segments=aplu_segments)(y)
        return y
    return f

def make_dense(size, activation=None, pieces=5, batch_norm=False, alpha=None, beta=None, share_params=False, blu_reg=0, aplu_segments=3, **kwargs):
    if activation != 'maxout':
        pieces = 1

    def f(x):
        y = [Dense(size, **kwargs)(x) for _ in range(pieces)]
        if batch_norm:
            y = [BatchNormalization()(p) for p in y]
        y = y[0] if len(y) == 1 else y
        if activation:
            y = make_activation(activation, alpha=alpha, beta=beta, blu_reg=blu_reg, aplu_segments=aplu_segments)(y)
        return y
    return f

def make_dropout(p, activation=None):
    def f(x):
        if activation == 'selu':
            y = AlphaDropout(p)(x)
        else:
            y = Dropout(p)(x)
        return y
    return f
