from pydub import AudioSegment
import os


def convert_audio_files(input_folder, output_format):
    """
    Converts all audio files in a specified folder to a specified audio format.

    Parameters:
    - input_folder (str): Path to the folder containing the audio files.
    - output_format (str): The desired audio format to convert the files to (ie, 'mp3', 'wav', 'ogg', etc).
    """

    output_format = output_format.lower().lstrip('.')
    input_folder = os.path.abspath(input_folder)


    for filename in os.listdir(input_folder):
        if not filename.endswith(output_format):
            original_path = os.path.join(input_folder, filename)

            try:
                sound = AudioSegment.from_file(original_path)
                new_file = os.path.splitext(original_path)[0] + '.' + output_format
                new_path = os.path.join(input_folder, new_file)

                sound.export(new_path, format=output_format)
                os.remove(original_path)
                print(f'Converted {filename} to {output_format}')
            except Exception as e:
                print(f'Error converting {filename}: {e}')


if __name__ == '__main__':
    input_folder = 'test_path'
    output_format = 'mp3'
    convert_audio_files(input_folder, output_format)