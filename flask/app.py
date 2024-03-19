from flask import Flask, request, render_template, jsonify
from pytube import YouTube
import librosa
import soundfile as sf
from datetime import datetime
import shutil
import os

app = Flask(__name__)

TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

@app.route('/')
def index():
    return render_template('index.html')

# Sample url: https://www.youtube.com/watch?v=ZPGi2yBqdqw
@app.route('/process-audio', methods=['POST'])
def process_audio():
    video_url = request.form.get('input_data_field')
    start_time = float(request.form.get('start_time', 0))
    end_time = float(request.form.get('end_time', 60))

    if not video_url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        yt = YouTube(video_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        download_path = audio_stream.download(output_path=TEMP_DIR)

        processed_audio_path = process_audio_file(download_path, start_time=start_time, end_time=end_time)

        model_prediction = "Placeholder prediction"  # placeholder for actual model prediction

        # Clean up the TEMP_DIR after processing
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR)

        return jsonify({"prediction": model_prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

def process_audio_file(file_path, target_sr=22050, target_format='wav', start_time=0, end_time=60):
    print(f'Processing {os.path.basename(file_path)}')

    # convert mp4 to target format
    y, sr = librosa.load(file_path, sr=target_sr)

    print(f'Original audio length: {len(y)/sr} seconds')
    
    # Calculate start and end frames for cutting
    start_frame = int(start_time * sr)
    end_frame = int(end_time * sr)
    y_cut = y[start_frame:end_frame]

    print(f'Cut audio length: {len(y_cut)/sr} seconds')
    # Construct a unique filename for the processed audio to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_filename = os.path.basename(file_path).rsplit('.', 1)[0]  # Removes extension
    processed_filename = f"{base_filename}_{timestamp}_cut.{target_format}"
    processed_audio_path = os.path.join(TEMP_DIR, processed_filename)

    print(f'Writing processed audio to {processed_audio_path}')
    # Write the processed (and optionally cut) audio to a new file
    sf.write(processed_audio_path, y_cut, sr, format=target_format)

    return processed_audio_path


if __name__ == '__main__':
    app.run(debug=True)
