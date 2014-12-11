MNIST Ruby Test
===

Testing classification of [MNIST](http://yann.lecun.com/exdb/mnist/) digits in Ruby.

Includes a Sinatra app that uses a trained ruby-fann neural network to predict digits drawn on a <canvas> element. The neural network was trained on all 60,000 training examples with 1 hidden layer of 300 neurons, and successfully classified ~93% of the test set.

MNIST data files not included in this repo.
