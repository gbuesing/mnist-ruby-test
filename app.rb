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
    pixels = get_pixels params[:dataURL]
    predict = $fann.run pixels
    decode_prediction(predict).to_json
  end

  private
    def get_pixels dataURL
      normalize = -> (val, fromLow, fromHigh, toLow, toHigh) {  (val - fromLow) * (toHigh - toLow) / (fromHigh - fromLow).to_f }

      canvas = ChunkyPNG::Canvas.from_data_url(dataURL)
      canvas.resample_bilinear!(28,28)

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