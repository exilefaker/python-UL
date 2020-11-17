import numpy as np
import matplotlib.pyplot as plt
import struct
from scipy.special import expit, softmax
from PIL import Image

from utilities import shuffle_arrays
from IO import load_mnist


class RBM(object):

    def __init__(self, layer_sizes=[784,500,500,2000], labels=10, alpha=0.1, weightcost=0.0001, momentum=0, useprobs='both', CD=1):
        """Defines a Restricted Boltzmann Machine - here, stochastic binary units are assumed
        layers - list of layer sizes, beginning with input layer and ending with the last hidden layer
        epochs - number of training sweeps through the data
        labels - number of labels/categories associated with labeled data (0 omits labels)
        alpha - learning rate
        weightcost - 'lambda' parameter used for L2 regularization
        momentum - momentum parameter for SGD with momentum"""

        ## Hyperparameters ##
        self.L = len(layer_sizes) #number of layers
        self.layer_sizes = layer_sizes #size of each layer
        self.num_labels = labels #number of labels

        self.alpha = alpha
        self.weightcost = weightcost
        self.momentum = momentum
        self.useprobs = useprobs
        self.CD = CD

        ## Parameters and gradients ##
        self.W = [] #weights for each layer
        self.d_W = [] #gradients for weights
        self.b = [] #biases for each layer
        self.d_b = [] #bias gradients
        for l in range(self.L):
            if l > 0:
                self.W.append((np.random.rand(layer_sizes[l],layer_sizes[l-1])*0.01 - 0.005)*np.sqrt(1/layer_sizes[l-1])) #Add weight matrix of shape (out, in)
                self.d_W.append(np.zeros(self.W[-1].shape))

            self.b.append(np.zeros(layer_sizes[l])) #Add bias vector of same dimensionality as layer
            self.d_b.append(np.zeros(self.b[-1].shape))
        if labels:
            self.W.append(np.random.rand(layer_sizes[self.L-1],self.num_labels)*0.01 - 0.005)
            self.d_W.append(np.zeros(self.W[-1].shape))
            self.b.append(np.zeros(self.num_labels))
            self.d_b.append(np.zeros(self.num_labels))


    def infer(self, source, weights, bias):  # return probabilities and sampled binary vector given an input vector 
        z = np.dot(weights,source)                              # compute input to a layer
        prob = expit(z + bias)                                  # sigmoid activation function for next layer to compute probabilities
        state = (np.random.rand(weights.shape[0]) <= prob) // 1  # sample from this distribution
        return prob, state


    def train(self, X, labels, config):

        # Helper functions
        def _grad(update_term,d):  # compute update
            return self.momentum*d + self.alpha * update_term

        def gradient(pos, neg, d):  # compute update term
            return _grad(pos - neg,d)

        def gradient_L2(param, pos, neg, d):  # compute update term with L2 regularization
            return _grad(pos - neg - (param*self.weightcost),d)

        def plog(s):  # print to screen and to file if logging enabled
            print(s)
            if logmode:
                g.write(s+"\n")

        ## INITIALIZE ################################################################################################|
        log, session_path = config.startup()  #IO operations - creates config.txt and folders for saved data

        N = X.shape[0]
        a, gen, q, p, lab_gen, lab_p = dict(), dict(), dict(), dict(), dict(), dict() # dummy variables for hidden states, fantasies, and probabilities
        up = {0: a, 1: q}  # dict to select states or probabilities for forward pass 
        down = {0: gen, 1: p}  # dict to select states or probs for downward pass
        label_down = {0: lab_gen, 1: lab_p}
        infer_with_probs = 0 if self.useprobs == ('none' or 'learn') else 1  # helper variables to set learning mode
        learn_with_probs = 0 if self.useprobs == ('none' or 'infer') else 1
        logmode = False

        if 'l' in config.save:
            logmode = True
            g = open(session_path + "/training_log.txt","w")  # File to log training progress

        ## MASTER LOOP ## - trains hidden layers up to L (no labels) or L - 1 (if labels)#############################|

        for l in range(config.train_from,self.L):
            errs = []
            k = 0
            plog("Training hidden layer %s\n------------------------" % l)
            #start epoch>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>|
            while k < config.epochs:
                shuffle_arrays(X,labels)  # Randomize training set order
                total_err = 0
                iter_err = 0
                err = 0
                i = 0

                plog("epoch %s:\n" % k)
                #loop through training set * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *|
                while i < N:

                    ## Positive phase ##------------------------------------------------------------------------------|
                    a[0] = X[i].astype('int')  # fetch image
                    q[0] = expit(a[0]-128)
                    a[0] = (np.random.rand(a[0].shape[0]) <= q[0]) / 1 

                    #a[0] = (a[0] >= (np.add(np.zeros(a[0].size),128)))/1  # binarize
                    z2 = 0  # reset placeholder for labels
                    alt = 0 # reset alternating Gibbs sampling counter

                    for layer in range(l):

                        if (layer == self.L-2 and self.num_labels):
                            labs = np.zeros(self.num_labels)
                            labs[labels[i]] = 1  # fetch label and one-hot encode
                            z2 = np.dot(self.W[self.L-1],labs)

                        q[layer+1], a[layer+1] = self.infer(q[layer], self.W[layer],(self.b[layer+1]+z2)) # compute layerth hidden layer probs and activation

                    ## Negative phase ##------------------------------------------------------------------------------|
                    #gen[l] = a[l]  # begin CD-n loop with data-driven hidden states
                    q_pos = q[l]

                    while alt < self.CD:  # alternating Gibbs sampling for n=CD steps
                        z2 = 0  # reset label input
                        p[l-1], gen[l-1] = self.infer(up[alt != 0 and infer_with_probs==1][l],np.transpose(self.W[l-1]),self.b[l-1])  # generate reconstruction on layer l-1

                        if (l == self.L-1 and self.num_labels):  # label units
                            lab_p[0], _ = self.infer(a[self.L-1],np.transpose(self.W[self.L-1]),self.b[self.L])  # generate label
                            lab_gen[0] = (lab_p[0] >= np.max(lab_p[0]))//1
                            z2 = np.dot(self.W[self.L-1],label_down[infer_with_probs][0])  # get negative-phase input from labels for top layer

                        q[l], a[l] = self.infer(down[infer_with_probs][l-1],self.W[l-1],self.b[l]+z2)  # bounce back up for CD learning
                        alt += 1

                    for reyal in range(l-1,0,-1):  # compute fantasies to measure reconstruction error
                        p[reyal-1], gen[reyal-1] = self.infer(down[infer_with_probs][reyal],np.transpose(self.W[reyal-1]),self.b[reyal-1])

                    ## COST + parameter updates ##====================================================================|
                    err = (sum(a[0]-gen[0]))**2  # sum errors
                    iter_err += err  # add to running total since last display
                    total_err += err  # add to running total for epoch

                    #CD-1 learning
                    pos_W = np.outer(q_pos,up[learn_with_probs][l-1])  # positive phase statistics
                    neg_W = np.outer(q[l],down[learn_with_probs][l-1])  # negative phase statistics

                    self.d_W[l-1] = gradient_L2(self.W[l-1],pos_W,neg_W, self.d_W[l-1])  # get weight updates (with regularization)
                    self.d_b[l] = gradient(q_pos,q[l],self.d_b[l])  # get bias updates
                    self.W[l-1] += self.d_W[l-1]  # do the updates
                    self.b[l] += self.d_b[l]
                    if (l == 1): 
                        self.d_b[l-1] = gradient(a[l-1],p[l-1],self.d_b[l-1])  # biases for lower layer
                        self.b[l-1] += self.d_b[l-1]

                    if (l == self.L-1 and self.num_labels):  # updates for label layer
                        pos_l = np.outer(q_pos,labs)
                        neg_l = np.outer(q[l],lab_gen[0])  #! since labels use a softmax unit, 'useprobs' arg doesn't apply
                        self.d_W[self.L-1] = gradient_L2(self.W[self.L-1],pos_l,neg_l,self.d_W[self.L-1])
                        self.d_b[self.L] = gradient(labs,lab_gen[0],self.d_b[self.L])
                        self.W[self.L-1] += self.d_W[self.L-1]
                        self.b[self.L] += self.d_b[self.L]            

                    ## SAVE every Xth iteration = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =|
                    if (i % config.save_freq == 0):
                        print("Saving data...")
                        #save data/fantasy pair - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -|
                        if 's' in config.save:
                            x_img = np.array((1-p[0])*255,dtype='uint8')
                            img = Image.fromarray(np.reshape(x_img,(28,28)),'L')
                            F_text = '(%s)' % lab_gen[0].argmax() if (self.num_labels > 0 and l == self.L-1) else ''
                            img.save(session_path + '/samples/layer%s/l%s-epoch%s-image%s-F%s.png' % (l,l,k,i,F_text))

                            img = Image.fromarray(np.reshape(255-X[i],(28,28)),'L')
                            D_text = '(%s)' % labels[i] if (self.num_labels > 0 and l == self.L-1) else ''
                            img.save(session_path + '/samples/layer%s/l%s-epoch%s-image%s-D%s.png' % (l,l,k,i,D_text))

                        #save parameters - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - |
                        if 'p' in config.save:
                            W_save = np.reshape(self.W[l-1],(self.layer_sizes[l]*self.layer_sizes[l-1]))
                            s = struct.pack('f'*(self.layer_sizes[l]*self.layer_sizes[l-1]), *W_save)
                            if (l == 1):
                                u = struct.pack('f'*(self.layer_sizes[l-1]), *self.b[l-1])
                            t = struct.pack('f'*self.layer_sizes[l], *self.b[l])
                            current_path = session_path + '/parameters/layer%s/l%s-epoch%s-iter%s' % (l,l,k,i)
                            with open(current_path,'wb') as f:
                                f.write(s)
                                if (l == 1):
                                    f.write(u)
                                f.write(t)
                                if (self.num_labels and l == self.L-1):
                                    W_save = np.reshape(self.W[self.L-1],(self.layer_sizes[l]*self.num_labels))
                                    s = struct.pack('f'*(self.layer_sizes[l]*self.num_labels), *W_save)
                                    t = struct.pack('f'*self.num_labels, *self.b[self.L])
                                    f.write(s)
                                    f.write(t)

                        #save filter images - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -|
                        #For higher hidden layers, depict filters in image space rather than displaying weights
                        if 'f' in config.save:
                            W_img = np.array([np.array(((1-expit(self.W[0][row,:]))*255)) for row in range(self.layer_sizes[1])])
                            for layer in range(1,l):
                                W_img = np.matmul(softmax(self.W[layer],axis=1),W_img)
                            for row in range(self.layer_sizes[l]):
                                W_img = W_img.astype('uint8')
                                Image.fromarray(np.reshape(W_img[row,:],(28,28)),"L").save(session_path + '/filters/layer%s/filter%s-epoch%s-iter%s.png' % (l,row,k,i))
                        #save plots - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - |
                        if i > 0 and 'c' in config.save:
                            plt.plot([x*500 for x in range((i+(k*N))//500)],errs, color="#6c3376", linewidth=3)
                            plt.xlabel('Iterations')
                            plt.ylabel('Reconstruction error')
                            plt.savefig(session_path + '/curves/layer%s/cost-l%s-iter%s.png' % (l,l,i+(k*N)))
                            plt.clf()

                            plt.hist(np.reshape(self.W[l-1],self.layer_sizes[l]*self.layer_sizes[l-1]),bins=50,color="#FF0000")
                            plt.xlabel('Values of parameters in W%s' % (l-1))
                            plt.ylabel('# of parameters')
                            plt.savefig(session_path + '/curves/layer%s/hist_W%s-iter%s.png' % (l,l-1,i+(k*N)))
                            plt.clf()

                            plt.hist(np.reshape(self.d_W[l-1],self.layer_sizes[l]*self.layer_sizes[l-1]),bins=200,color="#FF0000")
                            plt.xlabel('Values of gradients in d_W%s' % (l-1))
                            plt.ylabel('# of gradients')
                            plt.savefig(session_path + '/curves/layer%s/hist_d_W%s-iter%s.png' % (l,l-1,i+(k*N)))
                            plt.clf()

                            plt.hist(self.b[l],bins=30,color="#00AA00")
                            plt.xlabel('Values of parameters in b%s' % l)
                            plt.ylabel('# of parameters')
                            plt.savefig(session_path + '/curves/layer%s/hist_b%s-iter%s.png' % (l,l,i+(k*N)))
                            plt.clf()

                            plt.hist(self.d_b[l],bins=100,color="#00AA00")
                            plt.xlabel('Values of gradients in d_b%s' % l)
                            plt.ylabel('# of gradients')
                            plt.savefig(session_path + '/curves/layer%s/hist_d_b%s-iter%s.png' % (l,l,i+(k*N)))
                            plt.clf()

                            if (l == 1):
                                plt.hist(self.b[l-1],bins=30,color="#AA0000")
                                plt.savefig(session_path + '/curves/layer%s/hist_b%s-iter%s.png' % (l,l-1,i+(k*N)))
                                plt.xlabel('Values of parameters in b%s' % (l-1))
                                plt.ylabel('# of parameters')
                                plt.clf()

                                plt.hist(self.d_b[l-1],bins=100,color="#AA0000")
                                plt.savefig(session_path + '/curves/layer%s/hist_d_b%s-iter%s.png' % (l,l-1,i+(k*N)))
                                plt.xlabel('Values of gradients in d_b%s' % (l-1))
                                plt.ylabel('# of gradients')
                                plt.clf()

                    # end iteration * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * |
                    i += 1
                    if (i % 500 == 0):  # Print and log reconsruction error every (500) iterations
                        ierr = iter_err/500
                        plog("iter %s, cost: %s" % (i, ierr))
                        errs.append(ierr)
                        iter_err = 0
                # end epoch<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<|
                plog("cost after epoch %s: %s\n" % (k, (total_err/N)))
                k += 1

    def test(self, X, labels, config):

        def plog(s):  # print to screen and to file if logging enabled
            print(s)
            if logmode:
                g.write(s+"\n")

        _, session_path = config.startup()  #IO operations - creates config.txt and folders for saved data

        N = X.shape[0]
        a, gen, q, p, lab_p = dict(), dict(), dict(), dict(), dict() # dummy variables for hidden states, fantasies, and probabilities
        test_results = np.zeros(N,dtype='int')  # stores predictions

        logmode = False

        if 'l' in config.save:
            logmode = True
            g = open(session_path + "/test_log.txt","w")  # File to log training progress

        i = 0
        while i < N:
            lab_hist = np.zeros(self.num_labels) # init variable to store frequency of label units

            #Input test case
            a[0] = X[i].astype('int')  # fetch image
            q[0] = expit(a[0]-128)
            a[0] = (np.random.rand(a[0].shape[0]) <= q[0]) / 1 

            alt = 0 # reset alternating Gibbs sampling counter

            # Feed forward to obtain hidden states for hidden layer L-1
            for layer in range(self.L-1):
                q[layer+1], a[layer+1] = self.infer(q[layer], self.W[layer],self.b[layer+1]) # compute layerth hidden layer probs and activation

            while alt < self.CD:  # alternating Gibbs sampling for n=CD steps, holding inferred hidden units for images fixed
                lab_p[0], _ = self.infer(q[self.L-1],np.transpose(self.W[self.L-1]),self.b[self.L])
                lab_hist += (lab_p[0] >= np.max(lab_p[0]))//1
                z = np.dot(self.W[self.L-1],lab_p[0]) # Just use probabilities, since we're not learning here

                if alt < (self.CD-1):
                    q[self.L-1], a[self.L-1] = self.infer(q[self.L-2],self.W[self.L-2],self.b[self.L-1]+z)  # bounce back up (unless on last iteration of Gibbs sampling)

                alt += 1

            test_results[i] = np.argmax(lab_hist)  # Determine winner (if tie, smallest index is returned--fix this somehow)
            plog("Predicted/true class for example %s: %s/%s" % (i, test_results[i], labels[i]))

            i += 1

        test_score = np.sum([a == p for a,p in zip(labels,test_results)])/N
        plog('\nClassification accuracy: %s' % test_score)


    def sample(self, config, init_mode='random', X=None):

        _, session_path = config.startup()  #IO operations - creates config.txt and folders for saved data

        a, gen, q, p, lab_gen, lab_p = dict(), dict(), dict(), dict(), dict(), dict() # dummy variables for hidden states, fantasies, and probabilities
        if config.sample_class:
            num_classes = len(config.sample_class)

        i = 0
        while i < config.num_samples:

            if config.init_mode=='random':
                q[0] = np.random.rand(self.layer_sizes[0])

            elif config.init_mode=='MNIST_test':
                ind = np.random.randint(X.shape[0])
                a[0] = X[ind].astype('int')  # fetch image
                q[0] = expit(a[0]-128)
                x_img = np.array((1-q[0])*255,dtype='uint8')
                img = Image.fromarray(np.reshape(x_img,(28,28)),'L')
                img.save(session_path + '/seed%s.png' % i)

            alt = 0 # reset alternating Gibbs sampling counter
            #p[self.L-2] = q[self.L-2] = np.random.rand(self.b[self.L-2].shape)

            # Feed forward to obtain hidden states for hidden layer L-1
            for layer in range(self.L-2):
                q[layer+1], a[layer+1] = self.infer(q[layer], self.W[layer],self.b[layer+1]) # compute layerth hidden layer probs and activation

            #gen[self.L-2] = (np.random.rand(self.b[self.L-2].shape[0]) <= 0.5 // 1)  # sample state of penultimate hidden layer using biases
            lab_p[0] = np.zeros(self.num_labels)

            if config.sample_class:  # fix label units, if applicable
                sampled_class = np.random.randint(num_classes)
                lab_p[0][config.sample_class[sampled_class]] = 1
            else:
                #lab_p[0] = expit(self.b[self.L])
                #lab_gen[0] = (np.random.rand(self.b[self.L].shape[0]) <= lab_p[0] // 1)
                sampled_class = np.random.randint(self.num_labels)
                lab_p[0][sampled_class] = 1

            lab_gen[0] = lab_p[0]

            z = np.dot(self.W[self.L-1],lab_p[0])

            while alt < self.CD:  # alternating Gibbs sampling for n=CD steps -- just use probabilities, since we're not learning here
                q[self.L-1], a[self.L-1] = self.infer(q[self.L-2],self.W[self.L-2],self.b[self.L-1]+z)  # sample state of top layer
                p[self.L-2], gen[self.L-2] = self.infer(q[self.L-1],np.transpose(self.W[self.L-2]),self.b[self.L-2])  # generate reconstruction on penultimate HL
                if not config.sample_class:
                    lab_p[0], _ = self.infer(a[self.L-1],np.transpose(self.W[self.L-1]),self.b[self.L])  # generate label, if labels not clamped
                    lab_gen[0] = (lab_p[0] >= np.max(lab_p[0]))//1
                z = np.dot(self.W[self.L-1],lab_gen[0])

                alt += 1

            for reyal in range(self.L-2,0,-1):  # compute sample (fantasy)
                p[reyal-1], gen[reyal-1] = self.infer(p[reyal],np.transpose(self.W[reyal-1]),self.b[reyal-1])

            print("Saving sample %s..." % i)  # Save sample, obviously
            x_img = np.array((1-p[0])*255,dtype='uint8')
            img = Image.fromarray(np.reshape(x_img,(28,28)),'L')
            img.save(session_path + '/sample%s(%s).png' % (i,np.argmax(lab_gen[0])))

            i += 1





