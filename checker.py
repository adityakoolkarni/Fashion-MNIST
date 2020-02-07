# This specifies the number of layers and number of hidden neurons in each layer.
layer_specs: [784, 50, 10]

# Type of non-linear activation function to be used for the layers.
activation: "tanh"

# The learning rate to be used for training.
learning_rate: 0.005

# Number of training samples per batch to be passed to network
batch_size: 128

# Number of epochs to train the model
epochs: 50

# Flag to enable early stopping
early_stop: True

# History for early stopping. Wait for this many epochs to check validation loss / accuracy.
early_stop_epoch: 5

# Regularization constant
L2_penalty: 0

# Use momentum for training
momentum: False

# Value for the parameter 'gamma' in momentum
momentum_gamma: 0.9

# Mode to run the neural network in
mode: "Train"
