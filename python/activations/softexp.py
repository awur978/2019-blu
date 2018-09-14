from keras import backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
from keras.initializers import RandomUniform
from constraints import WeightClip

# 'Rectified' SoftExp

class SoftExp(Layer):
    def __init__(self, alpha_initializer='zeros', shared_axes=None, **kwargs):
        super(SoftExp, self).__init__(**kwargs)
        self.alpha_initializer = alpha_initializer
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        super(SoftExp, self).build(input_shape)

        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1

        self.alpha = self.add_weight(
            name='alpha',
            shape=input_shape[1:],
            initializer=self.alpha_initializer,
            constraint=WeightClip(-1, 1),
            trainable=True)

        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)

    def call(self, x):
        neg_alpha = K.relu(-self.alpha) + K.epsilon()
        pos_alpha = K.relu(self.alpha) + K.epsilon()
        pos_x = K.relu(x) + K.epsilon()
        log = K.log(neg_alpha*pos_x + 1) / neg_alpha
        exp = (K.exp(pos_alpha*pos_x) - 1) / pos_alpha
        return log + exp

    def compute_output_shape(self, input_shape):
        return input_shape
