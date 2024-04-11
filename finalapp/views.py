from django.shortcuts import render,redirect
from django.http import HttpResponse
from .utils import download_youtube_video,extract_text_from_video,translate_and_convert_to_audio, generate_html
import pymongo
#import webbrowser
import os
import time

connection = pymongo.MongoClient("mongodb://localhost:27017")
db = connection["verbavox"]
collection = db["files"]


def landing(request):
    return render(request,'landing.html')


def link(request):
    return render(request,'link.html')

def load(request):
    return render(request,'load.html')

def play(request):
    return render(request, 'play.html')






#main
def process_youtube_video(request):
     
     
    if request.method == 'POST':
# Step 1: Enter the YouTube video URL
     youtube_url=request.POST["youtube_url"]

# Step 2: Get user input for the desired filename (including extension)
     user_input_filename=request.POST["user_filename"]
     
#database insertion
     data = {"file_name":user_input_filename ,"link":youtube_url}
     insert = collection.insert_one(data) 
     print("successfully inserted to db")    

# Step 3: Download YouTube video
     output_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static', 'media')
  # You can customize the output directory
     success, video_file_path = download_youtube_video(youtube_url, output_directory, user_input_filename)

     if success:
        # Optional: Print the full path to the downloaded video
        video_file_path = os.path.abspath(video_file_path)
        print("Full Path to Video File:", video_file_path)
        # Step 4: Extract audio and text from video
        video_path = video_file_path
        extracted_text_file = extract_text_from_video(video_path)
        # Step 5: Translate text and convert to audio
        audio_path = translate_and_convert_to_audio(extracted_text_file, video_path)

        # Step 6: Generate HTML for video playback
        video_path = generate_html(video_path,audio_path)

        #webbrowser.open(f'file://{temp_html_path}', new=2)

     
        time.sleep(2)
        return render(request, 'play.html' , {'video_path': video_path})
     
    else:
        error_message  = "Failed to process!"
        return render(request, 'play.html', {'error_message': error_message})


def show_db(request): #function for  showing all files in database

    dblist = collection.find()
    return render(request, 'showdb.html' ,{'dblist': dblist})

    
     



# working
#loading working perfectly!!!
