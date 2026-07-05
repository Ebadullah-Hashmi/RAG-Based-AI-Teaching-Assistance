# How to use this RAG AI Teaching assistant on your own data     

## Step 1:- Collect Your Videos

-> Move all your video files to the video folder 

## Step 2:- Convert to mp3

-> Convert all the video files into mp3 by runing video_to_mp3

## Step 3:- Covert mp3 to Json

-> Convert all mp3 files to json by running mp3_to_json

## Step 4:- Convert the json file to vectors

-> Use the file preprocess_json to convert the json file to a dataframe with Embeddings
-> Save it as a joblib pickle

## Step 5:- Prompt generation and feeding to LLM

-> Read the joblib file and load it into the memory. Then create a relevent prompt as per the user query and feed it to the LLM