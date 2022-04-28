# modified from https://github.com/stamatelou/twitter_sentiment_analysis/blob/master/sentiment_analysis.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from textblob import TextBlob

def preprocessing(lines):
    words = lines.select(explode(split(lines.value, "\s+")).alias("word"))
    words = words.na.replace('', None)
    words = words.na.drop()
    words = words.withColumn('word', F.regexp_replace('word', r'http\S+', ''))
    words = words.withColumn('word', F.regexp_replace('word', '@\w+', ''))
    words = words.withColumn('word', F.regexp_replace('word', '#', ''))
    words = words.withColumn('word', F.regexp_replace('word', 'RT', ''))
    words = words.withColumn('word', F.regexp_replace('word', ':', ''))
    return words

def get_hashtags(lines):
    # split each tweet into words
    words = lines.select(explode(split(lines.value, "\s")).alias("word")).where('word like "#%"')    
    print(words)
    hashtags = words.withColumn('word', F.regexp_replace('word', '#', ''))   
    return hashtags

# text classification
def polarity_detection(text):
    return TextBlob(text).sentiment.polarity
def subjectivity_detection(text):
    return TextBlob(text).sentiment.subjectivity
def text_classification(words):
    # polarity detection
    polarity_detection_udf = udf(polarity_detection, StringType())
    words = words.withColumn("polarity", polarity_detection_udf("word"))
    # subjectivity detection
    subjectivity_detection_udf = udf(subjectivity_detection, StringType())
    words = words.withColumn("subjectivity", subjectivity_detection_udf("word"))
    return words

spark = SparkSession \
    .builder \
    .appName("sparkStream") \
    .getOrCreate()

# Create DataFrame representing the stream of input lines from connection to localhost:5555
lines = spark \
    .readStream \
    .format("socket") \
    .option("host", "127.0.0.1") \
    .option("port", 5555) \
    .load()

words = get_hashtags(lines)
#words = preprocessing(lines)

# text classification to define polarity and subjectivity
#words = text_classification(words)
#print("words", words)

#words = words.repartition(1)
words = words.coalesce(1)

# Generate running word count
wordCounts = words.groupBy("word").count().orderBy('count', ascending=False)


#wordCounts = words.groupBy("word").count()
# Start running the query that prints the running counts to the console
query = words.writeStream.queryName("all_tweets")\
        .outputMode("append").format("parquet")\
        .option("path", "./parc")\
        .option("checkpointLocation", "./check")\
        .trigger(processingTime='300 seconds').start()
query.awaitTermination()



query2 = wordCounts \
    .writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

query2.awaitTermination()

