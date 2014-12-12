#!/usr/bin/env ruby

require 'rubygems'
require 'bundler/setup'
require 'libsvm' # gem install rb-libsvm
require './mnist_loader'
require './cropper'

train_size = 10_000
test_size = 1000

puts "Loading training data..."
x_train, y_train = MnistLoader.training_set.get_data_and_labels train_size
x_train.map! {|row| Cropper.new(row).random_square_crop(24).to_a.flatten }
x_train.map! {|ary| Libsvm::Node.features(ary) }
decode_output = -> (output) { (0..9).max_by {|i| output[i]} }
y_train = y_train.map {|arr| decode_output.(arr)}

problem = Libsvm::Problem.new
parameter = Libsvm::SvmParameter.new

parameter.cache_size = 10 # in megabytes

parameter.eps = 0.001
parameter.gamma = 0.01
parameter.c = 10
parameter.kernel_type = Libsvm::KernelType::LINEAR

problem.set_examples(y_train, x_train)

puts "Training SVM with #{train_size} examples..."
t = Time.now
model = Libsvm::Model.train(problem, parameter)
puts "Training time: #{(Time.now - t).round(1)}s"

puts "\nLoading test data..."
x_test, y_test = MnistLoader.test_set.get_data_and_labels test_size
x_test.map! {|row| Cropper.new(row).random_square_crop(24).to_a.flatten }
x_test.map! {|ary| Libsvm::Node.features(ary) }
y_test = y_test.map {|arr| decode_output.(arr)}

error_rate = -> (errors, total) { ((errors / total.to_f) * 100).round }

mse = -> (actual, ideal) {
  (actual - ideal)**2
}

prediction_success = -> (actual, ideal) { decode_output.(actual) == decode_output.(ideal) }

run_test = -> (model, inputs, expected_outputs) {
  success, failure, errsum = 0,0,0
  inputs.each.with_index do |input, i|
    output = model.predict input
    output == expected_outputs[i] ? success += 1 : failure += 1
    errsum += mse.(output, expected_outputs[i])
  end
  [success, failure, errsum / inputs.length.to_f]
}

puts "Testing the trained network with #{test_size} examples..."

success, failure, avg_mse = run_test.(model, x_test, y_test)

puts "Trained classification success: #{success}, failure: #{failure} (classification error: #{error_rate.(failure, x_test.length)}%, mse: #{(avg_mse * 100).round(2)}%)"

filename = "data/trained_svm_#{train_size}_#{error_rate.(failure, x_test.length)}.net"
puts "Saving SVM to file: #{filename}"
model.save(filename)

