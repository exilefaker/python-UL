import argparse
from models import RBM
from session import Session
from IO import load_mnist

## DEFINE PARSER ##
parser = argparse.ArgumentParser(description='Train, validate/test, and sample from neural networks; save and load trained models and output detailed training logs.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Paths for saving, loading, and data retrieval
parser.add_argument('--dataset', type=str, metavar='STR', default='MNIST', help='string specifying dataset (for now, just load MNIST)')
parser.add_argument('--save_path', type=str, metavar='STR', default='save', help='directory to save data (data differs depending on mode and other parameters)')
parser.add_argument('--save_freq', type=int, metavar='INT', default=10000, help='number of iterations between checkpoints')
parser.add_argument('--save', nargs='*', metavar='STR',type=str, default=['c','p','f','s'], help="""list of strings specifying what to save:
                                                                         'l': training or test log
                                                                         'c': curves (error & other plots)
                                                                         'p': parameters
                                                                         'f': filter visualizations
                                                                         's': samples from the model
                                                                         Note: values other than 'l' ignored for tests; this argument is ignored in sample mode.""")
parser.add_argument('--init_from', type=str, metavar='STR', default=None, help="""use model parameters stored at specified path
                                                                   --ensure that network architectures match
                                                                   --loads most recently saved file at path
                                                                   --loads saved parameters for all layers, if they exist
                                                                   --required for 'test' and 'sample' modes""")
parser.add_argument('--train_from', type=int, metavar='INT', default=1, help='specify which layer to begin training on')
# Hyperparameters
parser.add_argument('--model', type=str, metavar='MODEL', default='RBM', help='currently only RBM')
parser.add_argument('--layer_sizes', nargs="+", type=int, metavar='INT', default=[784,500,500,2000], help='specify number of layers and size of each, beginning with input layer')
parser.add_argument('--labels', type=int, metavar='INT', default=10, help="""number of categories if labeled data; entering "0" bypasses labels
                                                            label layer attaches to final hidden layer""")
parser.add_argument('--epochs', type=int, metavar='INT', default=20, help='number of epochs (sweeps through data)')
parser.add_argument('--alpha', type=float, metavar='FLOAT', default=0.05, help='learning rate')
parser.add_argument('--L2_param', type=float, metavar='FLOAT', default=1e-4, help='L2 regularization parameter')
parser.add_argument('--momentum', type=float, metavar='INT', default=0, help='Momentum parameter for SGD with momentum')

parser.add_argument('--use_probs', type=str, metavar='STR', default='both', help="Use probabilities rather than states to 'infer' (negative phase only), 'learn', 'both', or 'none'")
parser.add_argument('--CD', type = int, metavar='INT', default=1, help="Number of steps of Gibbs sampling to use for Contrastive Divergence learning")

# Mode
parser.add_argument('--mode', type=str, metavar='STR', default='train', help='train, test, or sample')
parser.add_argument('--num_samples', type=int, metavar='INT', default=10, help="specify number of samples to generate (for use with 'sample' mode)")

args = parser.parse_args()

# Check for acceptable argument values
assert args.save_freq > 0, "--save_freq argument: Save frequency must be an integer greater than 0. To turn off logging, enter '--save'"
assert args.mode in ['train','test','sample'], "--mode argument: Please select 'train', 'test', or 'sample'"
if args.mode in ['train','test']:
    assert args.dataset, "Cannot train/test without a dataset."
if args.mode in ['test','sample']:
    assert args.init_from, "Cannot test or sample without a trained model."
assert set(args.save) == set(args.save) & set(['l','c','p','f','s']), "--save argument: Valid arguments are 'l', 'c', 'p', 'f', and/or 's"
if args.dataset=='MNIST':
    assert args.labels in [10, 0], "Training/testing on MNIST dataset requires 10 label units"
assert len(args.layer_sizes) > 1, "--layer_sizes argument: Need at least two layers"
assert args.use_probs in ['infer', 'learn', 'both','none'], "--use_probs argument: Must specify 'infer', 'learn', 'both', or 'none'"


# Execute program
if __name__ == '__main__':
    
    model = {'RBM': RBM}
    network = model[args.model](args.layer_sizes, args.labels, args.alpha, args.L2_param, args.momentum, args.use_probs, args.CD)
    config = Session(network, args.model, args.mode, args.dataset, args.save, args.save_path, args.save_freq, args.init_from, args.train_from, args.epochs, args.num_samples)
    mode = {'train': network.train, 'test': network.test, 'sample': network.sample}
    f = mode[args.mode]

    if args.mode in ['train','test']:
        if (args.dataset == 'MNIST'):
            X, labels = load_mnist(args.mode)

        f(X, labels, config)
    else:
        f(config)

