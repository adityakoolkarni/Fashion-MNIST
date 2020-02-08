################################################################################
################################################################################

import os, gzip,sys
import yaml
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(img):
    """
    Normalize your inputs here and return them.
    """
    assert isinstance(img, np.ndarray)
   
    normalized_image  = (img - np.min(img))/(np.max(img) - np.min(img))
    
    return normalized_image


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    assert isinstance(labels, np.ndarray)
    assert isinstance(num_classes, int)
    
    
    one_hot_labels = np.zeros((labels.shape[0], 10))
    for i, idx in enumerate(labels):
        one_hot_labels[i,idx] = 1
        
    return one_hot_labels


def display_image(img, label='None'):
    """ Display the input image and optionally save as a PNG.
    
    Args:
        img: The NumPy array or image to display
    
    Returns: None
    """
    # Convert img to PIL Image object (if it's an ndarray)
    if type(img) == np.ndarray:
        img = img.reshape((28,28))*256
        print("Converting from array to PIL Image")
        img = Image.fromarray(img)
    
    # Display the image
    img.show(title=label)

def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """
    labels_path = os.path.join(path, '{}-labels-idx1-ubyte.gz'.format(mode))
    images_path = os.path.join(path, '{}-images-idx3-ubyte.gz'.format(mode))

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels    = one_hot_encoding(labels, num_classes=10)
    
    return normalized_images, one_hot_labels


def softmax(x):
    """
    Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    try:
        x = x - np.max(x, axis=1, keepdims=True)
        assert x.any()<20
        softmax_out = np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)

    except OverflowError:
        print("Inputs to softmax too large")
        
    #raise NotImplementedError("Softmax not implemented")
    return softmax_out

def grad_softmax(y, targets):
    """
    Implement the gradient of the softmax function here.
    Remember to take care of the overflow condition.
    """
    grad_softmax = np.zeros((y.shape))
    try:
        grad_softmax = (targets - y)
            
    except OverflowError:
        print("Inputs to softmax too large")

    return grad_softmax


class Activation():
    """
    This class implements different types of activation functions for
    your neural network layers.
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError("{} is not implemented.".format(activation_type))

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a,test_mode = False):#Dummy test_mode
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a): 
        """
        Compute the forward pass.
        """
        self.x = a
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta, lr=0.005, momentum=0.005, reg=0):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid(self.x)

        elif self.activation_type == "tanh":
            grad = self.grad_tanh(self.x)

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU(self.x)

        return grad*delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        sigmoid_out = 1/(1+np.exp(-x))
        return sigmoid_out

    def tanh(self, x):
        """
        Implement tanh here.
        """

        tanh_out = np.tanh(x)
        
        return tanh_out

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        relu_out = np.copy(x)
        relu_out[relu_out<0] = 0
        
        return relu_out

    def grad_sigmoid(self,x):
        """
        Compute the gradient for sigmoid here.
        """
        grad_sigmoid_out = self.sigmoid(x)*(1-self.sigmoid(x))
        
        return grad_sigmoid_out

    def grad_tanh(self, x):
        """
        Compute the gradient for tanh here.
        """
        grad_tanh_out = (1-(self.tanh(x))**2)
        return grad_tanh_out

    def grad_ReLU(self, x):
        """
        Compute the gradient for ReLU here.
        """
        grad_relu_out = (x > 0) * 1
        
        return grad_relu_out

class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = np.random.randn(in_units, out_units)
        self.b = np.zeros(out_units)

        self.bst_wghts = self.w
        self.bst_b = self.b

        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)
        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this
        self.v_w = np.zeros((self.w.shape))
        self.v_b = np.zeros((self.b.shape))

    def __call__(self, x, test_mode = False):
        """
        Make layer callable.
        """

        if(test_mode):
            return self.forward_test_data(x)
        else:    
            return self.forward(x)

    def forward_test_data(self,x):
        """
        This executes only when test is being run
        Uses best weights
        """
        self.x = x
        self.a = x @ self.bst_wghts + self.bst_b
        
        return self.a

    def forward(self, x, test_mode = False):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        self.x = x
        self.a = np.dot(x, self.w) + self.b
        
        return self.a

    def backward(self, delta, lr=0.005, momentum=0, reg=0):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        
        self.d_w = np.matmul(self.x.T, delta)/config['batch_size']
        self.d_x = np.matmul(delta, self.w.T) #/config['batch_size']
        self.d_b = np.sum(delta, axis=0)
        
        # Weight Update 
        if momentum == 0:
            self.w += lr*self.d_w
            self.b += lr*self.d_b 
            
        else:
            self.d_w = self.d_w - reg*self.w# /config['batch_size']
            self.v_w = momentum*self.v_w + lr * self.d_w
            self.v_b = momentum*self.v_b + lr * self.d_b
            self.w += self.v_w 
            self.b += self.v_b
        return self.d_x


class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = Neuralnetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None, reg=0, test_mode = False):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets, reg, test_mode)

    def save_wghts(self):
        '''
        Saves the current weights in the forward layer when called
        '''
        for each_layer in self.layers:
            if(hasattr(each_layer,'w')):
                each_layer.bst_wghts = each_layer.w
                each_layer.bst_b = each_layer.b

    def forward(self, x, targets=None, reg=0, test_mode = False):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        for each_layer in self.layers:
            output = each_layer(x,test_mode)
            x = output
        self.y = softmax(x)
        self.targets = targets
        if targets is not None:
            if test_mode == True:
                accuracy = threshold_output(self.y,targets) 
                return  accuracy, self.y, self.loss(self.y, targets, reg)
            return self.y, self.loss(self.y, targets, reg)
        else:
            return self.y    

    def loss(self, logits, targets, reg=0):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        loss = 0
        for each_layer in self.layers:
            if(reg == 0):
                break
            if(hasattr(each_layer,'w')):
                loss = loss + reg*(np.sum(np.square(each_layer.w))) / (2 * targets.shape[0])
        
        loss = loss - np.sum(targets * np.log(logits))/targets.shape[0]
        return loss

    def backward(self, lr = 0.005, momentum=0, reg=0):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        delta = grad_softmax(self.y, self.targets)
        for each_layer in self.layers[::-1]:
            delta = each_layer.backward(delta, lr, momentum, reg)

def threshold_output(output,target):
    """
    Takes in input of softmax output and returns one hot encoded 
    output which has highes probability
    """

    encoded = np.eye(output.shape[1])
    max_prob = np.argmax(output,axis=1)
    corr_output = np.argmax(target,axis=1)

    accuracy = np.sum(max_prob == corr_output) / output.shape[0]
    return accuracy 


def train(net, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    assert x_train.shape[1] == 784
    assert y_train.shape[1] == 10

    train_loss = np.zeros(config['epochs'])
    val_loss = np.zeros(config['epochs'])
    train_true_predictions = np.zeros(config['epochs'])
    valid_true_predictions = np.zeros(config['epochs'])
    
    #net = Neuralnetwork(config)
    shuf_index = np.arange(x_train.shape[0])
    early_stp_cntr = 0
    min_reached = False
    
    # Train and validation runs 
    for n in tqdm(range(config['epochs'])):
        #np.random.shuffle(shuf_index)
        #x_train = x_train[shuf_index,:]
        #y_train = y_train[shuf_index,:]
        
        num_iters = 0
        min_loss = np.inf
        for i in range(0, x_train.shape[0], config['batch_size']):
            x_train_batch = x_train[i:i+config['batch_size']]
            y_train_batch = y_train[i:i+config['batch_size']]

            output, loss = net(x_train_batch, y_train_batch, config['L2_penalty'],test_mode = False)
            
            train_true_predictions[n] += threshold_output(output, y_train_batch)
            if config['momentum']:
                if config['L2_penalty'] > 0 :
                    net.backward(config['learning_rate'], config['momentum_gamma'], config['L2_penalty'])
                else: 
                    net.backward(config['learning_rate'], config['momentum_gamma'])
            else:
                if config['L2_penalty'] > 0 :
                    net.backward(config['learning_rate'], 0, config['L2_penalty'])
                else: 
                    net.backward(config['learning_rate'])
            num_iters += 1

            train_loss[n] += loss 
        train_loss[n] = train_loss[n] / num_iters #Average over the number of iterations
        train_true_predictions[n] = train_true_predictions[n] / num_iters
        assert train_true_predictions[n] <= 1

        num_iters = 0
        
        # Validation run
        for i in range(0, x_valid.shape[0], config['batch_size']):
            x_valid_batch = x_valid[i:i+config['batch_size']]
            y_valid_batch = y_valid[i:i+config['batch_size']]
            output, loss = net(x_valid_batch, y_valid_batch)
            valid_true_predictions[n] += threshold_output(output,y_valid_batch)
            val_loss[n] += loss  
            num_iters += 1

        val_loss[n] = val_loss[n] / num_iters
        valid_true_predictions[n] = valid_true_predictions[n] / num_iters
        assert valid_true_predictions[n] <= 1
        
        # Early stopping 
        if(val_loss[n] < min_loss):
            min_loss = val_loss[n]
            net.save_wghts()
        else:
            early_stp_cntr += 1

        if(early_stp_cntr == config['early_stop_epoch']):
            break
    plt.figure(1)
    plt.plot(np.arange(config['epochs']), train_loss, label='Train')
    plt.plot(np.arange(config['epochs']), val_loss, label='Validation')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    
    # Plot the train and validation Accuracy against the number of epochs
    plt.figure(2)
    plt.plot(np.arange(config['epochs']), train_true_predictions, label='Train Accuracy')
    plt.plot(np.arange(config['epochs']), valid_true_predictions, label='Validation Accuracy')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """
    
    accuracy, sft_max_out_y, loss =  model(X_test, y_test, reg= config['L2_penalty'], test_mode = True)
    print("Test Accuracy is",accuracy)




if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")

    x_valid = x_train[50000:60000]
    y_valid = y_train[50000:60000]
    x_train = x_train[:50000]
    y_train = y_train[:50000]

    # Train the model
    train(model, x_train, y_train, x_valid, y_valid, config)

    # Test the model
    test(model, x_test, y_test)
