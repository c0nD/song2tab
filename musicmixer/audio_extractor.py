import zipfile
import os

def extract_audio(zip_path, extract_to):
    """
    Extracts audio from a zip file.

    Parameters:
    - zip_path (str): Path to the zip file.
    - extract_to (str): Path to extract the audio to.
    """

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f'Extracted audio to {extract_to}')


if __name__ == '__main__':
    zip_path = 'test_set.zip'
    extract_to = 'test_path'
    extract_audio(zip_path, extract_to)