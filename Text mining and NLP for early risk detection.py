#!/usr/bin/env python
# coding: utf-8


# In[215]:
#Importing libraries

#General libraries
import numpy as np
import pandas as pd
import sys
import os 
os.environ['PYTHONHASHSEED'] = "0" #making sure it hashes everytime the same thing

#Libraries for parsing and getting text from websites
from codecs import xmlcharrefreplace_errors
import feedparser
import hashlib
import urllib.parse
import requests
from bs4 import BeautifulSoup
import ssl

#Libraries for NLP
from gensim.parsing.preprocessing import strip_tags
from deep_translator import GoogleTranslator
import re
from transformers import pipeline
import spacy
import spacy_transformers
from spacy.cli import download
from gensim.parsing.preprocessing import stem_text
import country_converter as coco
download("en_core_web_trf")

#Libraries for SQL / Database loading
from sqlalchemy import create_engine
import psycopg2




# In[216]:
#Loading extras

#Loading extras for parsing
ssl._create_default_https_context = ssl._create_unverified_context #avoiding SSL errors
headers =  {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0"} #avoiding some bot-shields

#Loading csv as dictionairy to match inhabitant names to country names 
path = "/home/valentin_werner/data/demonyms.csv"
denonym = pd.read_csv(path, header = None, index_col = 0).T.to_dict("records")[0]

# Loading country csv
path_count = r"/home/valentin_werner/data/countries.csv"
cand_labels = pd.read_csv(path_count)["en"].to_numpy()

#Loading extras for the SQL connection
conn_path = r"/home/valentin_werner/data/conn.txt"
conn = pd.read_csv(conn_path, header =None)[0][0]
engine = create_engine(conn)


# In[217]:
#Functions to retrieve information from websites
def search_rss(link, key_search, count):
    '''searches for the key of newest feed of a given link'''
    feed = feedparser.parse(link)
    entry = feed.entries[count]
    return entry[key_search]

def scrape_text(link, name, attrs, number):
    """retrieves text from the respective article; input is based on data which is received from rss feed"""
    response = requests.get(link, headers = headers) 
    soup = BeautifulSoup(response.content, "html.parser")
    content = soup.find_all(name, attrs)[number]
    return content.text

def scrape_bsi_text(website):
    """special scraping function which was designed specifically for the bsi website, having a very different source code"""
    response = requests.get(website, headers = headers)
    soup = BeautifulSoup(response.content, "html.parser")
    infocol1 = [info.text for info in soup.find_all("span", attrs={"class":"infocol-1"})]
    infocol2 = [info.text for info in soup.find_all("span", attrs={"class":"infocol-2"})]
    desc = str(soup.find("meta", attrs = {"property":"og:description"}))
    string = desc[:len(desc)-29]
    string = string[15:]
    return " \n ".join([" ".join([infocol1[index], i]) for index, i in enumerate(infocol2)]) + " \n " + string


# In[220]:
#Functions to make sure every website is only scraped once
def create_id(link):
    """creates hash for every link;
    as such this will be a unique identifier and serve as prime key in the database"""
    hash_link = hashlib.md5(link.encode()).hexdigest()
    return str(hash_link)

def check_if_exists(article_id):
    """queries all article_ids (the hashes from the links) from the database 
    and checks if the hash for the website to be scraped already exists"""
    sql = 'SELECT article_id FROM it_security.articles'
    article_ids = pd.read_sql(sql=sql,con=engine)
    hashed = list(article_ids['article_id'])
    if article_id in hashed:
        return True #True means article was already hashed
    else:
        return False 

# In[358]:

#Function to combine scraping functions
def get_data(feed, count):
    """gets information from RSS feed and scrapes text if the article isnt existing yet"""
    if 'published' in feedparser.parse(feed).entries[count].keys():
        date = search_rss(feed, "published", count)
    else:
        date = search_rss(feed, "updated", count)
    
    website = search_rss(feed, "link", count)
    title = search_rss(feed, "title", count)
    parsed_url = urllib.parse.urlparse(website).netloc
    feed_id = create_id(website)
    
    # Check if exists
    check = check_if_exists(feed_id)
    if check == False:  #True = does already exist
        #scrape text
        if parsed_url in cs_dictionary.keys():
            text = strip_tags(scrape_text(website, cs_dictionary[parsed_url][0], cs_dictionary[parsed_url][1],cs_dictionary[parsed_url][2])).replace('\n',' ')
            return np.array([feed_id, title, date, website, text])
        elif parsed_url == 'www.bsi.bund.de': #seperate scraping function for bsi
            text = GoogleTranslator(source='auto', target='en').translate(text=strip_tags(scrape_bsi_text(website)).replace('\n',' ')) #translate to english 
            return np.array([feed_id, title, date, website, text])
    else: return None #skip if already exists

# In[296]:
#Function writing security tags
def sec_tags_write(text, fk, ner):
    """takes the data from the scraped text and retrieves IT Security buzzwords based on a pretrained hugginface transformer;
    requires scraped text, foreign key (hashed article url) and ner model (huggingface; cyner)"""

    #intialize list for results
    mylist = []
    #get all relevant words from the model
    result = ner(text)
    if len(result) > 0:
        #the results are often split into syllables / parts and need to be reunited
        for index, ent in enumerate(result):
            #if the first syllable is tagged, we always append it
            if index == 0: 
                mylist.append(ent["word"])
            #if we find a syllable which is directly after another tagged syllable, we assume they belone together 
            #and append it to the same string as the syllable that was already appended
            elif ent["index"] == result[index-1]["index"] + 1:
                mylist[-1] = mylist[-1] + ent["word"] 
            #if we find a syllable that was tagged that is not following another tagged syllable we append it
            else: mylist.append(ent["word"])
        #as the syllables are split with a specific symbol, we need to clear the symbol so the words are logical
        clean_list = []
        for ent in mylist: 
            clean_list.append(re.sub("▁", " ", ent).lstrip().rstrip()) #note: this is no underscore
        #turn the cleaned tag list together with the foreign key (being the hashed article url) into a dataframe
        df_sec = {"security_tag":list(set(clean_list)),
            "article_id": fk}
        df_sec_tags = pd.DataFrame(df_sec)
        #push the dataframe to the relational database into the security_tags table
        df_sec_tags.to_sql("security_tags", engine, schema = "it_security", if_exists = "append", index= False)
    else: #in case there are no tags found we still want to show that there are no tags
        df_sec = {"security_tag": ["tagless"],
            "article_id": fk}
        df_sec_tags = pd.DataFrame(df_sec)
        df_sec_tags.to_sql("security_tags", engine, schema = "it_security", if_exists = "append", index= False)

# In[]:
#Functions preparing geo_tags
def stem_unique(geotags):
    """stems geological words; such as Germans -> German"""
    newlist = list(set([stem_text(word) for word in geotags])) #stems and removes duplicates
    return newlist

def map_coun_names(word):
    '''checks the existence of a geological word in the given dictionary of nationalities and countries'''
    try: 
        return denonym[word.capitalize()]
    except:
        #if it doesnt exist, we return it unprocessed; this makes sense in case it is already a processed word (e.g., Germany)
        return word 

def country_codes(word):
    '''checks the word and returns the country code'''
    if coco.convert(names=word, to='ISO2') == "not found":
        return None
    else:
        return coco.convert(names=word, to='ISO2')


# In[]:
#functions to write geo_tags
def geo_stem(text, nlp):
    """tags geographic entities and initiates the stemming on them; removing all duplicates along the way;
    requires scraped text and nlp model"""
    wanted_tags = ["NORP", "GPE", "LOC"] #the tags refer to nationalities, geo-political entities, locations
    geo_tags = []
    doc = nlp(str(text)) #retrieves entities for the scraped text
    for ent in doc.ents:
        if ent.label_ in wanted_tags: 
            geo_tags.append(ent.text) #we only take the geographically related labels
            print(ent.label_, ent.text)
    stemmed = stem_unique(list(set(geo_tags))) #remove duplicates from input and stem the geographic entities
    return stemmed

def geo_clean_and_write(stemmed_list, fk):
    """make geographical data uniform and write it into the database; 
    requires stemmed list and foreign key (=hash from url)"""
    countries = list(map(map_coun_names, stemmed_list)) #maps the stemmed geographic entities to country names (German -> Germany)
    #turn countries into ISO2 Code if it exists; this will drop words like "Berlin" etc.
    print(countries)
    try:
        cc = [i for i in [country_codes(country) for country in countries] if i] #maps to ISO code, also drops all None
        print(cc)
        if len(cc) >  0: #if there are any countries after the step before, they are turned into a dataframe 
            df_el = {"country_tag":cc,
                "article_id": fk}
            df_geo_tags = pd.DataFrame(df_el)
            #the countries are uploaded into the relational database tables country_tags with the foreign key (= hash from url)
            df_geo_tags.to_sql("country_tags", engine, schema = "it_security", if_exists = "append", index= False)
    except Exception:
        pass
    

# In[]:
#Writing meta data
def write_meta(data):
    """writes meta data to relational database; requires parsed infos from RSS feed"""
    dict_meta = {"article_id": [data[0]],
        "article_title": [data[1]],
        "article_date": [data[2]],
        "article_url": [data[3]],
        "article_text": [data[4]]}
    df_meta = pd.DataFrame(dict_meta)
    #writes the dict from the RSS data into the article table in the relational database
    df_meta.to_sql("articles", engine, schema = "it_security", if_exists = "append", index= False)




# In[341]:
#Dictionaries off RSS feeds and how to scrape related articles
#Dictionary on how to scrape individual feeds
cs_dictionary = {
    'us-cert.cisa.gov':['div',{"class":"field field--name-body field--type-text-with-summary field--label-hidden field--item"},4],
    'www.cisecurity.org':['div',{"class":"template-main-content"},0],
    'www.darkreading.com':['script', {"type":"application/ld+json"},0],
    'cyber.gc.ca':["div", {"class":"field field--name-body field--type-text-with-summary field--label-hidden field--item"},0],
    'www.techtarget.com':["section",{"id":"content-body"},0],
    'gbhackers.com':["div", {"class":"td-post-content"},0],
    "www.cshub.com":["div",{"mb-3"},0],
    "www.secureworks.com":["div",{"dashed"},0], 
    "www.welivesecurity.com":["div",{"col-md-10 col-sm-10 col-xs-12 formatted"},0],
}

#Dictionary of RSS feeds we are taking information from
feeds = [#"https://www.cisa.gov/uscert/ncas/current-activity.xml", #duplicated?
         "https://www.us-cert.gov/ncas/current-activity.xml",
         "https://www.cisecurity.org/feed/advisories",
         "https://www.cshub.com/rss/articles",
         "https://www.secureworks.com/rss?feed=blog",
         "https://www.welivesecurity.com/feed/",
         "https://www.bsi.bund.de/SiteGlobals/Functions/RSSFeed/RSSNewsfeed/RSSNewsfeed_WID.xml;jsessionid=61A22E4430B03697E67346435668A7AB.internet462",
         #"https://cyber.gc.ca/api/cccs/rss/v1/get?feed=alerts_advisories&lang=en",
         #"https://gbhackers.com/feed/",
         "https://www.techtarget.com/searchsecurity/rss/Security-Wire-Daily-News.xml"]
         #"https://www.darkreading.com/rss/all.xml"]

# In[359]:

def main(feeds = feeds):
    """main functions executing everything"""
    nlp = spacy.load("en_core_web_trf") #initialize the nlp model for stemming and geographical tags
    ner = pipeline("ner", model = "AI4Sec/cyner-xlm-roberta-base") #initialize the ner for cyber security tags

    for feed in feeds: #go through all feeds
        counter = 0
        while True:
            try:
                data = get_data(feed, counter) #get basic data from feed
                print(data)
                print(counter)
                if data is None: 
                    counter += 1
                    continue #go to next feed if there is no data 
                else: #if there is data received
                    write_meta(data) #write basic data to database; needs to be done first
                    geo_stemmed = geo_stem(data[4], nlp) #create tags and stem them
                    if geo_stemmed != []:
                        geo_clean_and_write(geo_stemmed, data[0]) #write ISO country codes into database
                    sec_tags_write(data[4], data[0], ner) #write security tags into database
                    counter += 1
            except IndexError: #if any information is missing when writing, skip this article entirely
                break



# In[]:
main() #execute it all in 6 symbols!


# %%
