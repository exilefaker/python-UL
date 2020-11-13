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

`python RBM_cmdline.py --parameter ARG`

Argument/parameter descriptions forthcoming.
