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
import functions_spotify
from functions_spotify import *
import torch
from pyspark.sql import SparkSession
#import psycopg2 as pg
#pg.extensions.register_type(pg.extensions.UNICODE)
#pg.extensions.register_type(pg.extensions.UNICODEARRAY)
print("All packages succesfully imported")

# code for retrieving all jsons

def rec_fold_trav(root_dir):

    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir) for f in filenames if os.path.splitext(f)[1] == '.json']

    return result


root_dir = "/home/moritzdeecke/project_spotify/transcripts"

all_jsons = rec_fold_trav(root_dir)

print(len(all_jsons))

print("All jsons ready")

# we instantiate a spark session

spark = SparkSession.builder.master("local[1]") \
    .appName("SparkByExamples.com").getOrCreate()

# we parallelize the process across all jsons

rdd=spark.sparkContext.parallelize(all_jsons)

rdd2=rdd.map(lambda x: main(x))

rdd2.collect()


