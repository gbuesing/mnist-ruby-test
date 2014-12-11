require 'rubygems'
require 'bundler/setup'
require 'narray'

class Cropper
  attr_reader :data, :dimension, :matrix

  def initialize data
    @data = data
    @dimension = Math.sqrt(@data.length).ceil
    @matrix = NMatrix[*@data.each_slice(@dimension).to_a]
  end

  def square_crop size, x_offset = 0, y_offset = 0
    x_range = Range.new x_offset, x_offset + size - 1
    y_range = Range.new y_offset, y_offset + size - 1
    @matrix[x_range, y_range]
  end

  def random_square_crop size 
    max_offset = dimension - size + 1
    square_crop size, rand(max_offset), rand(max_offset)
  end
end
