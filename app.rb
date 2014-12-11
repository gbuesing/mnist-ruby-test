require 'rubygems'
require 'bundler/setup'
require 'ruby-fann'
require 'sinatra/base'
require 'chunky_png'
require 'json'

$fann = RubyFann::Standard.new(:filename=>"./train/trained_nn_300_60000_6.net")


class DigitClassifierApp < Sinatra::Application
  get '/' do
    erb :home
  end

  post '/predict' do
    canvas = ChunkyPNG::Canvas.from_data_url(params[:dataURL])
    canvas.resample_bilinear!(28,28)
    pixels = get_normalized_pixels canvas
    predict = $fann.run pixels
    {predict: decode_prediction(predict), data_url: canvas.to_data_url}.to_json
  end

  private
    def get_normalized_pixels canvas
      normalize = -> (val, fromLow, fromHigh, toLow, toHigh) {  (val - fromLow) * (toHigh - toLow) / (fromHigh - fromLow).to_f }

      pixels = []
      28.times do |y| 
        28.times {|x| pixels << canvas[x, y] }
      end
      
      max, min = pixels.max, pixels.min
      pixels = pixels.map {|p| normalize.(p, min, max, 0, 1) }
      pixels
    end

    def decode_prediction result
      (0..9).max_by {|i| result[i]}
    end

end