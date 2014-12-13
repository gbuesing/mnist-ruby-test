require 'rubygems'
require 'bundler/setup'
require 'ruby-fann'
require 'sinatra/base'
require 'chunky_png'
require 'json'

$fann = RubyFann::Standard.new(:filename=>"./train/trained_nn_300_60000_7_crop.net")


class DigitClassifierApp < Sinatra::Application
  get '/' do
    erb :home
  end

  post '/predict' do
    canvas = ChunkyPNG::Canvas.from_data_url(params[:dataURL])
    # canvas.save 'input.png'
    canvas = center_and_downsample canvas
    # canvas.save 'cropped.png'
    random_cropped = 4.times.map { canvas.crop(rand(5), rand(5), 24, 24) }
    predict_sums = Array.new(10, 0)
    random_cropped.each do |cropped|
      pixels = get_normalized_pixels cropped
      predict = $fann.run pixels
      predict.each_with_index {|val, i| predict_sums[i] += val}
    end
    {predict: decode_prediction(predict_sums), data_url: canvas.to_data_url}.to_json
  end

  private
    def center_and_downsample canvas
      canvas.trim!
      size = [canvas.width, canvas.height].max
      square = ChunkyPNG::Canvas.new(size, size, ChunkyPNG::Color::TRANSPARENT)
      offset_x = ((size - canvas.width) / 2.0).floor
      offset_y = ((size - canvas.height) / 2.0).floor
      square.compose! canvas, offset_x, offset_y
      square.resample_bilinear!(20,20)
      square.border! 4, ChunkyPNG::Color::TRANSPARENT
      square
    end

    def get_normalized_pixels canvas
      normalize = -> (val, fromLow, fromHigh, toLow, toHigh) {  (val - fromLow) * (toHigh - toLow) / (fromHigh - fromLow).to_f }

      pixels = []
      24.times do |y| 
        24.times {|x| pixels << canvas[x, y] }
      end
      
      max, min = pixels.max, pixels.min
      pixels = pixels.map {|p| normalize.(p, min, max, 0, 1) }
      pixels
    end

    def decode_prediction result
      (0..9).max_by {|i| result[i]}
    end

end