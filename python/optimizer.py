from keras.optimizers import Adam, RMSprop, sgd

def make_optimizer(optimizer, momentum=0., **kwargs):
    if optimizer == 'adam':
        return Adam(**kwargs)
    elif optimizer == 'rmsprop':
        return RMSprop(**kwargs)
    elif optimizer == 'sgd':
        return sgd(momentum=momentum, nesterov=True, **kwargs)
    else:
        raise ValueError('invalid optimizer "{}"'.format(optimizer))
