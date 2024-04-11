#works
from pytube import YouTube
import os
import re
import os
from moviepy.editor import *
from pydub import AudioSegment, silence
import whisper
from transformers import  MBartForConditionalGeneration, MBart50TokenizerFast
import tempfile
from gtts import gTTS
from pathlib import Path

import torch
from torch.autograd import Variable


def load_model(model_path):
    # Load the trained SUNO/BARK model
    model = torch.load(model_path)
    model.eval()
    return model


def get_user_input_filename():
    """
    Prompt the user for a filename (including extension) and return the sanitized input.
    """
    user_input = input("Enter the desired filename (including extension): ")
    # Sanitize the input by removing special characters
    sanitized_filename = re.sub('[^a-zA-Z0-9\s.-]', '', user_input)
    return sanitized_filename




def download_youtube_video(url, output_path, user_input_filename):
    try:
        yt = YouTube(url)
        video_stream = yt.streams.filter(file_extension="mp4", res="720p").first()

        # Sanitize the user input filename by removing special characters
        sanitized_filename = re.sub('[^a-zA-Z0-9\s.-]', '', user_input_filename)

        # Use the user input filename for saving the video
        video_file_name = f"{sanitized_filename}.mp4"
        video_file_path = os.path.join(output_path, video_file_name)

        video_stream.download(output_path, filename=video_file_name)
        print("Video downloaded successfully.")
        return True, video_file_path
    except Exception as e:
        print("Error downloading YouTube video:", str(e))
        return False, None
    



def extract_text_from_video(video_path):
    try:
        # Load the video
        video_clip = VideoFileClip(video_path)

        # Extract audio from the video
        audio = video_clip.audio

        # Get the directory of the video file
        video_directory = os.path.dirname(video_path)

        # Save the audio as a temporary MP3 file in the video directory
        audio_path = os.path.join(video_directory, "temp_audio.mp3")
        audio.write_audiofile(audio_path, codec="mp3")

        print("Audio Path:", audio_path)

        # Check if the audio file exists
        if os.path.exists(audio_path):
            # Provide the full path to the Whisper model
            model=whisper.load_model('base')
            print(audio_path)
            result = model.transcribe(audio_path)

            # Save the transcription to a text file
            transcription_file_path = os.path.join(video_directory, "transcription.txt")
            with open(transcription_file_path, "w") as f:
                f.write(result["text"])

            print("Transcription complete. Transcription saved to:", transcription_file_path)
            return transcription_file_path
        else:
            print("Error: Audio file does not exist at the specified path.")
    except Exception as e:
        print("Error extracting text from video:", str(e))
    finally:
        # Close the audio and video clips
        audio.close()
        video_clip.close()






def translate_and_convert_to_audio(extracted_text_file, video_path):
    translated_text = translation(extracted_text_file)
    target_language = 'ta'

    # Use text to speech library to convert the translated text to audio
    tts = gTTS(text=translated_text, lang=target_language, slow=False)
    temp_audio_path = os.path.join(os.path.dirname(video_path), 'translated_audio.mp3')
    tts.save(temp_audio_path)

    return temp_audio_path




def transcribe_audio(audio_path, model):
    try:
        # Load the audio file into memory
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()

        # Prepare the audio data for the model
        audio_data = Variable(torch.from_numpy(audio_data).float()).unsqueeze(0)

        # Obtain the model's prediction for the audio data
        prediction = model(audio_data)

        # Convert the prediction to text
        transcript = prediction.to("cpu").detach().numpy().flatten().tolist()

        print("Transcript:", transcript)
        return True, transcript
    except Exception as e:
        print("Error transcribing audio:", str(e))
        return False, None





def generate_html(video_path, translated_audio_path):
    # Replace the audio of the video with the new translated audio
    new_video = replace_audio(video_path, translated_audio_path)

    # Save the new video with the updated audio to a temporary file
    temp_video_path = os.path.join(os.path.dirname(video_path), 'video_with_new_audio.mp4')
    new_video.write_videofile(temp_video_path)
     
    ''' # Generate HTML content for video playback with controls
    html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Video Player</title>
        </head>
        <body>
            <video width="100%" height="90%" controls>
                <source src="{temp_video_path}" type="video/mp4">
                Your browser does not support the video tag.
            </video>

             
        </body>
        </html>
    """

    # Save the HTML content to a temporary file
    temp_html_path = os.path.join(os.path.dirname(video_path), 'video_player.html')
    with open(temp_html_path, 'w') as html_file:
        html_file.write(html_content)'''

    return temp_video_path 


    

def replace_audio(video_path, translated_audio_path):
    # Load the video and audio
    video = VideoFileClip(video_path)
    audio = AudioFileClip(translated_audio_path)

    # Replace the audio of the video with the new translated audio
    new_video = video.set_audio(audio)

    return new_video




def translation(extracted_text_file):
    # Load the model and tokenizer
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

    # Read the content from the file
    with open(extracted_text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Translate each line and overwrite the content in the file
    with open(extracted_text_file, 'w', encoding='utf-8') as f:
        for line in lines:
            translated_text = translate_text(line, model, tokenizer) #to translate in chunks, the translation is seperated into two functions
            f.write(translated_text + '\n')
    return translated_text






def translate_text(text, model, tokenizer):
    max_chunk_size = 512

    # Split the input into chunks
    input_chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]

    # Translate each chunk
    translations = []
    for chunk in input_chunks:
        model_inputs = tokenizer(chunk, return_tensors="pt")
        generated_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.lang_code_to_id["ta_IN"])
        translation_chunk = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translations.extend(translation_chunk)

    # Concatenate the results
    translation = " ".join(translations)
    return translation








