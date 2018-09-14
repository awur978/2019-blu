from keras.regularizers import l2

from fitnet import make_fitnet
from shakeshake import make_shakeshake
from tiny import make_tiny
from wrn import make_wrn

def make_model(model, input_shape, output_size, args):
    if model == 'fitnet':
        return make_fitnet(args.activation,
                           input_shape,
                           output_size,
                           dropout=args.dropout,
                           batch_norm=args.batch_norm,
                           kernel_regularizer=l2(args.l2_regularization),
                           alpha=args.alpha,
                           beta=args.beta,
                           share_params=args.share_params,
                           blu_reg=args.blu_reg,
                           aplu_segments=args.aplu_segments)
    elif model == 'ss' or model == 'ssn':
        return make_shakeshake(args.activation,
                               input_shape,
                               output_size,
                               batch_norm=args.batch_norm,
                               kernel_regularizer=l2(args.l2_regularization),
                               alpha=args.alpha,
                               beta=args.beta,
                               share_params=args.share_params,
                               blu_reg=args.blu_reg,
                               depth=args.depth,
                               k=args.k,
                               residual=model == 'ss',
                               batch_size=args.batch_size,
                               aplu_segments=args.aplu_segments)
    elif model == 'tiny':
        return make_tiny(args.activation,
                         input_shape,
                         output_size,
                         dropout=args.dropout,
                         batch_norm=args.batch_norm,
                         kernel_regularizer=l2(args.l2_regularization),
                         alpha=args.alpha,
                         beta=args.beta,
                         share_params=args.share_params,
                         blu_reg=args.blu_reg,
                         aplu_segments=args.aplu_segments)
    elif model == 'wrn' or model == 'wnn':
        return make_wrn(args.activation,
                        input_shape,
                        output_size,
                        dropout=args.dropout,
                        batch_norm=args.batch_norm,
                        kernel_regularizer=l2(args.l2_regularization),
                        alpha=args.alpha,
                        beta=args.beta,
                        share_params=args.share_params,
                        blu_reg=args.blu_reg,
                        depth=args.depth,
                        k=args.k,
                        residual=model == 'wrn',
                        aplu_segments=args.aplu_segments)
    else:
        raise ValueError('invalid model "{}"'.format(model))

    return Model(inputs=x, outputs=y)

__all__ = ['make_fitnet', 'make_shakeshake', 'make_tiny', 'make_wrn']
