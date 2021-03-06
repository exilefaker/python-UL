import glob
import numpy as np
import struct
from os import path, mkdir


class Session(object):

    def __init__(self, model, modelname='RBM', mode='train', dataset='MNIST', save=['c','p','f','s'], save_path='save', save_freq=2000, init_from=None, train_from=1, epochs=20, num_samples=10, sample_class=None, init_mode='random'):

        # Check consistency of settings
        assert train_from < model.L, "Cannot begin training from hidden layer %s in a model with %s hidden layers." % (train_from, model.L-1)
       
        self.save = save
        self.save_path = save_path
        self.save_freq = save_freq
        self.init_from = init_from
        self.train_from = train_from
        self.epochs = epochs
        self.num_samples = num_samples
        self.sample_class = sample_class
        self.init_mode = init_mode

        self.model = model
        self.modelname = modelname
        self.mode = mode
        self.dataset = dataset

    def startup(self):

        def _plog(s):
            print(s)
            f.write(s+"\n")

        if self.init_from: # Load previously trained parameters--currently, loads all paramters stored for the model from most recent save state in path

            for l in range(1,self.model.L):
                layer_path = self.init_from + ('/parameters/layer%s/' % l)
                if path.exists(layer_path):
                    log_list = glob.glob('%s*' % layer_path)
                    if log_list:
                        latest = max(log_list, key=path.getctime)
                        print("Loading saved parameters for layer %s..." % l)
                        with open(latest,'rb') as f:
                            data_size = struct.calcsize('f'*self.model.layer_sizes[l]*self.model.layer_sizes[l-1])
                            self.model.W[l-1] = np.reshape(struct.unpack('f'*self.model.layer_sizes[l]*self.model.layer_sizes[l-1],f.read(data_size)),(self.model.layer_sizes[l],self.model.layer_sizes[l-1]))
                            if (l==1):
                                data_size = struct.calcsize('f'*self.model.layer_sizes[l-1])
                                self.model.b[l-1] = np.array(struct.unpack('f'*self.model.layer_sizes[l-1],f.read(data_size)))
                            data_size = struct.calcsize('f'*self.model.layer_sizes[l])
                            self.model.b[l] = np.array(struct.unpack('f'*self.model.layer_sizes[l],f.read(data_size)))
                            if (self.model.num_labels and l == self.model.L-1):
                                data_size = struct.calcsize('f'*self.model.layer_sizes[l]*self.model.num_labels)
                                self.model.W[self.model.L-1] = np.reshape(struct.unpack('f'*self.model.layer_sizes[l]*self.model.num_labels,f.read(data_size)),(self.model.layer_sizes[l],self.model.num_labels))
                                self.model.b[self.model.L] = np.array(struct.unpack('f'*self.model.num_labels,f.read()))
            print("")

        # Initialize session, update log.txt which just stores the last session ID#
        log = 0
        session_path = ''
        if self.save:
            fn = '/%s_log.txt' % self.mode
            if path.exists(self.save_path + fn):
                log = np.fromfile(self.save_path + fn, dtype='int', count=1)[0]
            log += 1
            if not path.exists(self.save_path):
                mkdir(self.save_path)
                print("\nCreating folder %s/ to save data..." % self.save_path)
            with open(self.save_path + fn, 'w') as f:
                np.array([log]).tofile(f)
            # Create required sub-directories if not found
            session_path = self.save_path + ('/%s_session' % self.mode) + str(log)
            if not path.exists(session_path):
                mkdir(session_path)
                print("Subfolder " + session_path + " created to store session info.")

        if self.mode == 'train':

            save_dict = {'c': 'curves', 'p': 'parameters', 'f': 'filters', 's': 'samples'}
            save_args = [x for x in self.save if x != 'l']
            save_args = [save_dict[key] for key in save_args]
            for subpath in save_args:
                if not path.exists(session_path+"/"+subpath):
                    mkdir(session_path+"/"+subpath)
                    print("Subfolder " + session_path+"/"+subpath + " created to store %s." % subpath)
                    for layer in range(self.train_from, self.model.L):
                        layer_path = session_path+"/"+subpath+"/layer%s" % layer
                        if not path.exists(layer_path):
                            mkdir(layer_path)
                            print("Creating %s subfolder for layer %s" % (subpath, layer))

            print("\nTraining %s (usng SGD) on dataset \'%s\' with hyperparameters:\n" % (self.modelname, self.dataset))

            with open(session_path+'/config.txt', 'w') as f:
                f.write("Config for %s training session" % self.modelname +str(log)+"\n\n")
                f.write("Dataset: %s" % self.dataset + "\n")
                s = "Input layer size: %s" % self.model.layer_sizes[0]
                _plog(s)
                for l in range(1,self.model.L):
                    s = "Hidden layer %s size: %s" % (l, self.model.layer_sizes[l])
                    _plog(s)
                if self.model.num_labels:
                    s = "Number of labeled categories: %s" % self.model.num_labels
                    _plog(s)
                if self.modelname == 'RBM':
                    s = "Using CD-%s learning" % self.model.CD
                    _plog(s)
                s = "Learning rate: %s" % self.model.alpha
                _plog(s)
                s = "L2 regularization parameter: %s" % self.model.weightcost
                _plog(s)
                s = "Momentum: %s" % self.model.momentum
                _plog(s)
                d = {'none': 'neither learning nor inference', 'both': 'both learning and inference', 'learn': 'learning', 'infer': 'inference'}
                s = "Using probabilities for %s \n" % d[self.model.useprobs]
                _plog(s)
                s = "For %s epochs\n" % self.epochs
                print("For %s epochs\n" % self.epochs)
                f.write("Epochs: %s" % self.epochs + "\n")
                if self.init_from:
                    s = "Beginning with the parameters stored in " + self.init_from
                    _plog(s)
                if self.train_from:
                    s = "Starting traning at layer %s" % self.train_from
                    _plog(s)

            if save_args:
                if len(save_args) == 1:
                    save_string = save_args[0]
                elif len(save_args) == 2:
                    save_string = '%s and %s' % tuple(save_args)
                elif len(save_args) > 2:
                    save_string = ', '.join(save_args[:-1]) + " and %s" % save_args[-1]
                print("Saving " +save_string+" every %s iterations" % self.save_freq)
            if 'l' in self.save:
                print("Maintaining training log")
            print("\n")

        elif self.mode == 'test':
            print("\nTesting the model stored at %s on the \'%s\' dataset, using %s steps of Gibbs sampling.\n" % (self.init_from, self.dataset, self.model.CD))

        elif self.mode == 'sample':
            print("\nDrawing %s samples from the model stored at %s, using %s step(s) of Gibbs sampling.\n" % (self.num_samples, self.init_from, self.model.CD))
            if self.sample_class:
                print("Label units clamped to class(es) %s" % self.sample_class)

        return log, session_path