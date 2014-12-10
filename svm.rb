#!/usr/bin/env ruby

require 'rubygems'
require 'libsvm' # gem install rb-libsvm
require './mnist_loader'

train_size = 30_000
test_size = 10_000

puts "Loading training data..."
x_train, y_train = MnistLoader.training_set.get_data_and_labels train_size


problem = Libsvm::Problem.new
parameter = Libsvm::SvmParameter.new

parameter.cache_size = 1 # in megabytes

parameter.eps = 0.001
parameter.c = 10

examples = x_train.map {|ary| Libsvm::Node.features(ary) }

problem.set_examples(y_train, examples)

model = Libsvm::Model.train(problem, parameter)

# pred = model.predict(Libsvm::Node.features(1, 1, 1))
# puts "Example [1, 1, 1] - Predicted #{pred}"

