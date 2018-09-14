from keras.datasets import cifar10, cifar100

def load_data(dataset):
    if dataset == 'cifar10':
        return cifar10.load_data() + (10,)
    elif dataset == 'cifar100':
        return cifar100.load_data(label_mode='fine') + (100,)
    else:
        raise ValueError('invalid dataset "{}"'.format(dataset))
