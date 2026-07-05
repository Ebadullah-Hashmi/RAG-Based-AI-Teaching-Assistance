#Converts videos to mp3
import os
import subprocess

files = os.listdir("videos")
for file in files:
    tutorial_num = file.split(" [")[0].split(" #")[1].split(".mp4")[0]
    file_name = file.split(" _ ")[0]
    print(tutorial_num,file_name)
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{tutorial_num}_{file_name}.mp3"])