#!/usr/bin/env ruby

require 'rubygems'
require 'bundler/setup'
require 'ruby-fann' # gem install ruby-fann
require './mnist_loader'

train_size = 20_000
test_size = 10_000
hidden_neurons = 30

puts "Loading training data..."
x_train, y_train = MnistLoader.training_set.get_data_and_labels train_size


train = RubyFann::TrainData.new(:inputs=> x_train, :desired_outputs=> y_train)
fann = RubyFann::Standard.new(:num_inputs=> 28*28, :hidden_neurons=> [hidden_neurons], :num_outputs=> 10)

puts "Training network with #{train_size} examples..."
t = Time.now

fann.train_on_data(train, 1000, 10, 0.01) # 1000 max_epochs, 10 errors between reports and 0.1 desired MSE (mean-squared-error)
puts "Training time: #{(Time.now - t).round(1)}s"


puts "\nLoading test data..."
x_test, y_test = MnistLoader.test_set.get_data_and_labels test_size

error_rate = -> (errors, total) { ((errors / total.to_f) * 100).round }

mse = -> (actual, ideal) {
  errors = actual.zip(ideal).map {|a, i| a - i }
  (errors.inject(0) {|sum, err| sum += err**2}) / errors.length.to_f
}

decode_output = -> (output) { (0..9).max_by {|i| output[i]} }
prediction_success = -> (actual, ideal) { decode_output.(actual) == decode_output.(ideal) }

run_test = -> (nn, inputs, expected_outputs) {
  success, failure, errsum = 0,0,0
  inputs.each.with_index do |input, i|
    output = nn.run input
    prediction_success.(output, expected_outputs[i]) ? success += 1 : failure += 1
    errsum += mse.(output, expected_outputs[i])
  end
  [success, failure, errsum / inputs.length.to_f]
}

puts "Testing the trained network with #{test_size} examples..."

success, failure, avg_mse = run_test.(fann, x_test, y_test)

puts "Trained classification success: #{success}, failure: #{failure} (classification error: #{error_rate.(failure, x_test.length)}%, mse: #{(avg_mse * 100).round(2)}%)"

filename = "data/trained_nn_#{hidden_neurons}_#{train_size}_#{error_rate.(failure, x_test.length)}.net"
puts "Saving neural network to file: #{filename}"
fann.save(filename)

