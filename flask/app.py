from flask import Flask, request, render_template, jsonify
from pytube import YouTube
import librosa
import soundfile as sf
import os
sys.path.append('Z:/song2tab/models/experimentation/experiments/tab_cnn/models/fold-3')


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process-audio', methods=['POST'])
def process_audio():
    # Retrieve the URL from the form submission
    video_url = request.form.get('video_url')

    if not video_url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        yt = YouTube(video_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        download_path = audio_stream.download(output_path='./temp')
        
        processed_file_path = process_audio_file(download_path)
        
        # model_prediction = your_model_prediction_function(processed_file_path)
        
        os.remove(download_path)
        os.remove(processed_file_path)

        # Placeholder for actual model prediction
        model_prediction = "Model prediction based on audio."
        
        return jsonify({"prediction": model_prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
def process_audio_file(file_path, target_sr=22050, target_format='wav'):
    audio, sr = librosa.load(file_path, sr=target_sr)
    new_file_path = file_path.rsplit('.', 1)[0] + '_processed.' + target_format
    sf.write(new_file_path, audio, sr)
    return new_file_path


if __name__ == '__main__':
    app.run(debug=True)