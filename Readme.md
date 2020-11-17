# Python command line tools for unsupervised learning (python-UL)
(Working title)

This is a set of command line tools for training, testing, and sampling from artificial neural networks, and for visualizing and logging details of the training process.

I am mainly developing this tool for autodidactic purposes, though in the end I hope to render it suitable for performing at least small-scale experiments for use in practical applications.

The intended focus is on unsupervised learning algorithms, e.g. generative-models-based approaches that learn to mimic the density of the training data:
* Restricted Boltzmann machines (RBMs) trained using Contrastive Divergence
* Deep Belief Networks (DBNs) composed of stacked RBMs
* Variational autoencoders (VAEs)
* The Helmholtz machine trained using the wake-sleep algorithm

Such discriminative tests as are included in this package can be run using joint density models of "data" and "labels", such as a DBN with two input channels (the model type around which this library was initially developed).

# Usage

```
usage: RBM_cmdline.py [-h] [--dataset STR] [--save_path STR] [--save_freq INT]
                      [--save [STR [STR ...]]] [--init_from STR] [--train_from INT] [--model MODEL]
                      [--layer_sizes INT [INT ...]] [--labels INT] [--epochs INT] [--alpha FLOAT]
                      [--L2_param FLOAT] [--momentum INT] [--anneal BOOL] [--use_probs STR] [--CD INT]
                      [--mode STR] [--num_samples INT] [--sample_class [INT [INT ...]]]
                      [--init_mode STR]

Train, validate/test, and sample from neural networks; save and load trained models and output
detailed training logs.

optional arguments:
  -h, --help            show this help message and exit
  --dataset STR         string specifying dataset (for now, just load MNIST) (default: MNIST)
  --save_path STR       directory to save data (data differs depending on mode and other parameters)
                        (default: save)
  --save_freq INT       number of iterations between checkpoints (default: 10000)
  --save [STR [STR ...]]
                        list of strings specifying what to save: 'l': training or test log 'c': curves
                        (error & other plots) 'p': parameters 'f': filter visualizations 's': samples
                        from the model Note: values other than 'l' ignored for tests; this argument is
                        ignored in sample mode. (default: ['c', 'p', 'f', 's'])
  --init_from STR       use model parameters stored at specified path -ensure that network
                        architectures match -loads most recently saved file at path -loads saved
                        parameters for all layers, if they exist -required for 'test' and 'sample'
                        modes (default: None)
  --train_from INT      specify which layer to begin training on (default: 1)
  --model MODEL         currently only RBM (default: RBM)
  --layer_sizes INT [INT ...]
                        specify number of layers and size of each, beginning with input layer
                        (default: [784, 500, 500, 2000])
  --labels INT          number of categories if labeled data; entering "0" bypasses labels label layer
                        attaches to final hidden layer (default: 10)
  --epochs INT          number of epochs (sweeps through data) (default: 20)
  --alpha FLOAT         learning rate (default: 0.05)
  --L2_param FLOAT      L2 regularization parameter (default: 0.0001)
  --momentum INT        Momentum parameter for SGD with momentum (default: 0)
  --anneal BOOL         Boolean to toggle annealing of learning rate [NOT YET IMPLEMENTED] (default:
                        True)
  --use_probs STR       Use probabilities rather than states to 'infer' (negative phase only),
                        'learn', 'both', or 'none' (default: both)
  --CD INT              Number of steps of Gibbs sampling to use for Contrastive Divergence learning
                        (default: 1)
  --mode STR            train, test, or sample (default: train)
  --num_samples INT     For 'sample' mode: specify number of samples to generate (default: 10)
  --sample_class [INT [INT ...]]
                        For 'sample' mode: specify which classes to generate from -if one value is
                        provided, all samples will be from the specified class -if more than one,
                        class for each sample is chosen from a flat distribution (default: None)
  --init_mode STR       specify data to use to initialize Gibbs sampling random: random vector in
                        [0,1] MNIST_test: random MNIST test image (default: random)
```

# Modes

`train`: Train a neural net on the specified dataset.
`test`: If applicable, test network on a discriminative task (using specified test set).
`sample`: Draw a sample from the distribution of a generative model.

# Models

For now, only one base model, the restricted Boltzmann machine (RBM), is implemented. RBMs are Markov random fields (undirected graphical models) partitioned into "visible" an "hidden" units, with no connections within these partitions. For use with energy-based unsupervised learning algorithms such as Contrastive Divergence. See: https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf

Stacked RBMs can be used to create a Deep Belief Network (DBN), which is a directed top-down generative model, in which the feedforward connections are used for approximate inference from data. Alternating Gibbs sampling at the top two layers can be used to run a Markov chain long enough to approach the network's equilibrium distribution, for testing or sampling purposes.

The tools here support RBMs/DBNs with two input "channels": images (or more generally data) and labels. These model types can be used to build joint density models of data and their labels.

# Datasets

So far, only the MNIST handwritten digit database is natively supported.

# Goals

The immediate goals of the project are to:
(a) Train an RBM on MNIST with discriminative performance >95%;
(b) Train a qualitatively decent generative model of MNIST digits;
(c) Once this benchmark is reached, implement the Variational Autoencoder (VAE).
