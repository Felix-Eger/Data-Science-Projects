# code for creating dataframe on word basis for each json

import os
#import sqlite3
import pandas as pd
import numpy as np
import glob
import pandas as pd
import numpy as np
import json
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
import math
from pathlib import Path
import pandas as pd
import numpy as np
import json
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
import math
from pathlib import Path
from sqlalchemy import create_engine
import torch

def create_time_col(col):
    '''takes a column as an input and coerces it to a numerical column'''
    
    return float(col[:-1])

def speaker_change(speaker, speaker_shift):
    '''takes speaker and speaker shifted as input and verifies the change of the speaker as binaries'''
    
    # if value is first value
    if np.isnan(speaker_shift) == True:
        return 1
    elif speaker != speaker_shift:
        return 1
    else:
        return 0

def get_word_df(path):
    '''gets the json as a path and calculates a word dataframe'''
    
    # coerce json into pandas df
    json_file = pd.read_json(path)
    
    # get index of last entry
    last_index = json_file.index[-1]
    
    # set empty dic
    empty_dic = {"startTime":[], "endTime":[], "word":[],"speakerTag":[]}
    
    # fill dic
    empty_list = [[empty_dic["startTime"].append(dic["startTime"]),empty_dic["endTime"].append(dic["endTime"]), empty_dic["word"].append(dic["word"]),empty_dic["speakerTag"].append(dic["speakerTag"])]  for dic in json_file.iloc[last_index]["results"]["alternatives"][0]["words"]]
    
    # create df
    df_words = pd.DataFrame(empty_dic)
    
    # coerce time values to float
    df_words["startTime"] = df_words.apply(lambda x: create_time_col(x["startTime"]), axis = 1)
    df_words["endTime"] = df_words.apply(lambda x: create_time_col(x["endTime"]), axis = 1)
    
    # calculate the difference between values
    df_words["timeDelta"] = df_words["endTime"] - df_words["startTime"]
    
    # shifts series
    df_words["speaker_shift"] = df_words.speakerTag.shift()
    
    # calculates the points where speaker changes as binaries
    df_words["speaker_shift_num"] = df_words.apply(lambda x: speaker_change(x["speakerTag"], x["speaker_shift"]), axis = 1)
    
    # calculates the cumulative sums for each speaker
    df_words["speakerPart"] = df_words.speaker_shift_num.cumsum()
    
    return df_words[["word", "speakerTag", "timeDelta", "speakerPart"]]


# now we aggregate across the words to get an aggregated view per part spoken by an individual person
def speaker_agg(table):
    '''creates the aggregated view on a speaker level'''
    
    # instantiate the classifier
    classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)

		# we aggregate per speaker part 
    df_agg = table.groupby("speakerPart").agg({'word': ' '.join, "speakerTag":"mean", "timeDelta":"sum", "speakerPart":"count"}).rename(columns= {"word":"content", "speakerTag":"speaker", "timeDelta":"time", "speakerPart":"count"})
    
		# we aggregate further per individual speaker based on the parts
    df_agg_speaker = df_agg.groupby("speaker").agg({"content":" ".join, "time":"sum", "count":"sum"})
    
    # we create equally sized sublists for each speaker
    sublists = [create_equal_sublists(speaker_con) for speaker_con in df_agg_speaker["content"]]

    # we run the classifier over each sublist
    scores = [classifier(substrings, return_all_scores = True) for substrings in  sublists]

    # we pass each score subset into the dataframe creation, creating an individual dataframe per speaker
    sub_dfs = [create_emotion_df(score).agg("mean") for score in scores]

    # we create a dataframe that contains the averaged
    em_df_total = pd.DataFrame(sub_dfs, index= df_agg_speaker.index)

    # we concat the dataframes
    result = pd.concat([df_agg_speaker, em_df_total], axis = 1)
    
    return result


# Missing: Sentiment on sliding window
def sliding_window(string):
    '''Seperate whole text into chunks of size 30'''
    
    # instantiate the classifier
    classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)

    # instantiate the summarizer 
    summarizer = pipeline("summarization", model="paulowoicho/t5-podcast-summarisation", tokenizer="paulowoicho/t5-podcast-summarisation")
    
    chunks = 30 

    # we split the strings into words of each chunks
    strings_split = string.split()

    # we split ech list into 30 equal chunks
    splitted_text = np.array_split(strings_split, 30)

    # we concatenate the strings back 
    chunks_joined = [" ".join(sublist) for sublist in splitted_text]

    summarized_chunks = summarizing(chunks_joined, summarizer)

    #print(summrized_chunks)

    # we call the equal sublist function on each sublist
    equ_sublists = [create_equal_sublists(chunk) for chunk in chunks_joined]

    print(np.shape(equ_sublists))

    # we calculate the scores for each sublist 
    sub_scores = [[classifier(chunk_split, return_all_scores = True)[0] for chunk_split in chunk] for chunk in equ_sublists]

    # we create averaged dataframe for each result returned within the windows
    em_sub = [create_emotion_df(window_scores).agg("mean").to_numpy() for window_scores in sub_scores]

    # we create a dataframe out of the results an return it 
    df_sliding = pd.DataFrame(em_sub, columns = ["sadness", "anger", "fear", "love", "joy", "surprise"])

    # we append the sliding integer info
    slid_win = np.arange(1,31,1)
    df_sliding["window"] = slid_win

    # we summarize the chunks
    sum_chunks = [dic["summary_text"] for dic in summarized_chunks]

    # we append the podcasts
    df_sliding["summary"] = sum_chunks

    return df_sliding


def create_equal_sublists(string):
    '''Check the size of the string and split into equal chunks when bigger 512'''
    
    # we split the string by 512 to observe the size of it
    split_len = math.ceil(len(string) / 512)

    # we evaluate whethetr it is bigger or smaller
    if split_len > 1:
      # we split the list into sublists for each word
      string_splitted = string.split()
      # we split the sublists into equal chunks
      splitted_text = np.array_split(string_splitted, split_len)
      # we concate each sublist back 
      list_concat = [" ".join(sublist) for sublist in splitted_text]
      # we return the sublists
      return list_concat
    else:
      return [string]

def create_emotion_df(scores):
    '''takes the unordered scores as an input and returns a dataframe with the results ordered'''

    # order to respect
    order = ["sadness", "anger", "fear", "love", "joy", "surprise"]
    order_scores = {"sadness": [], "anger": [], "fear": [], "love":[], "joy":[], "surprise": []}

    # accessing each list and ordering the scores of the sublists
    scores_ordered = [[order_scores[item["label"]].append(item["score"]) for item in sublist] for sublist in scores]

    df = pd.DataFrame(order_scores)

    return df


# Missing: Text Summary
def summarizing(input, summarizer):
    '''Summarizes a string input'''
    summary = summarizer([input[i] for i in range(0, len(input), 1)], min_length=5, max_length=30)
    return summary



# Missing: Main Function
def main(json):
    '''The main function called for each json'''

    # setting up database connection 
    conn = create_engine("postgresql://username:password!@ip-address:port/db_name")

    # stem episode id 
    episode_id = Path(json).stem

    # extract the content of a json
    df_json = get_word_df(json)

    # extract the df's on a speaker basis
    json_agg = speaker_agg(df_json)

    # we add speaker and episode id's
    json_agg["speaker_id"] = json_agg.index
    json_agg["episode_id"] = episode_id

  	# we rename the columns according to db and drop content
    json_agg = json_agg.rename(columns = {"anger":"sp_anger", "surprise":"sp_surprise", "fear":"sp_fear", "sadness":"sp_sadness", "time":"sp_time", "joy":"sp_joy", "love":"sp_love"}).drop("content", axis = 1)

    # get the total text
    total_text = " ".join(df_json["word"])

    # get the data on a window basis
    json_win = sliding_window(total_text)

    # we add episode_id 
    json_win["episode_id"] = episode_id

    # we rename the columns according to db
    json_win = json_win.rename(columns = {"window":"window_slide", "sadness":"su_sadness", "surprise":"su_surprise", "anger":"su_anger", "fear":"su_fear", "love":"su_love", "joy":"su_joy"})

    # sending both dfs to the database
    json_agg.to_sql('speaker', conn, if_exists = "append", index=False)
    json_win.to_sql('summary', conn, if_exists = "append", index=False)

    return None
