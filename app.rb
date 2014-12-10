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
      # canvas.grayscale!
      # canvas.trim!
      canvas.resample_bilinear!(28,28)
      # canvas.border! 4, 0
      # canvas.save 'test.png'

      newpng = ChunkyPNG::Image.new(28, 28)

      pixels = []
      28.times do |y| 
        28.times {|x| pixels << canvas[x, y] }
      end
      max, min = pixels.max, pixels.min
      pixels = pixels.map {|p| normalize.(p, min, max, 0, 1) }

      28.times do |y|
        x_offset = y * 28
        28.times {|x| newpng[x,y] = ChunkyPNG::Color("black @ #{pixels[x + x_offset]}") }
      end

      # newpng.save 'test2.png'


      pixels
    end

    def decode_prediction result
      (0..9).max_by {|i| result[i]}
    end

end