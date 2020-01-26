# *********** CONFIGURACIÓN DEL ENTORNO ************

import findspark
findspark.init()

import re
import os
import pandas as pd
from matplotlib import pyplot as plt
from math import floor

from pyspark import SparkConf, SparkContext, SQLContext, HiveContext
from pyspark.sql import Row
from pyspark.sql.functions import lit
from pyspark.sql.functions import date_format, hour, from_utc_timestamp

conf = SparkConf()
conf.setMaster("local")
conf.setAppName("myApp")
sc = SparkContext(conf=conf)

# ************ IMPORTAR DATOS ******************

# Cargamos los tweets en formato JSON desde nuestro fichero en HDFS

sqlContext = SQLContext(sc)
path_to_data = '/user/data/tweets28a_sample.json'
tweets_sample = sqlContext.read.json(path_to_data)

print("El dataset cargado contiene %d tweets" % tweets_sample.count())

# Estudiamos el schema de los datos importados

print ("\nShcema de los datos cargados:\n")
tweets_sample.printSchema()

print ("\nVisualización de los datos:\n")
tweets_sample.show()

# *************** VISUALIZACIÓN CON SQL ************************

sqlContext.sql('DROP TABLE IF EXISTS tweets_sample')
sqlContext.registerDataFrameAsTable(tweets_sample, "tweets_sample")

# Ver usuarios con más tweets, incluyendo información adicional 

users_agg = sqlContext.sql("SELECT user.screen_name, MAX(user.friends_count) AS friends_count, MAX(user.followers_count) AS followers_count, user.lang, COUNT(text) AS tweets FROM tweets_sample WHERE user.lang = 'es' GROUP BY user.screen_name, user.lang ORDER BY tweets DESC")
users_agg.show()

# Cargamos la visualización en una tabla

sqlContext.sql('DROP TABLE IF EXISTS user_agg')
sqlContext.registerDataFrameAsTable(users_agg, "user_agg")

# Estudiamos los usuarios que han recibido más retweets, visualizando otra información adicional y calculando el ration de retweets por tweets

retweeted = sqlContext.sql('SELECT COUNT(tweets_sample.text) AS retweeted, tweets_sample.retweeted_status.user.screen_name, MAX(user_agg.friends_count) AS friends_count, MAX(user_agg.followers_count) AS followers_count, MAX(user_agg.tweets) AS tweets, COUNT(tweets_sample.text)/MAX(user_agg.tweets) AS ratio_tweet_retweeted FROM user_agg INNER JOIN tweets_sample ON user_agg.screen_name=tweets_sample.retweeted_status.user.screen_name GROUP BY tweets_sample.retweeted_status.user.screen_name ORDER BY ratio_tweet_retweeted DESC')
retweeted.limit(20).show()

# *************** VISUALIZACIÓN CON DF ************************

# Usuarios con más tweets e información adicional

users = tweets_sample.where("user.lang == 'es'").select("user.screen_name", "user.friends_count", "user.followers_count", "text")

users_agg = users.groupBy("screen_name")\
                 .agg({"friends_count": "max", "followers_count": "max", "text": "count"})\
                 .orderBy("count(text)", ascending=False)

users_agg_new = users_agg.withColumnRenamed('max(friends_count)','friends_count')\
                         .withColumnRenamed('max(followers_count)','followers_count')\
                         .withColumnRenamed('count(text)','tweets')

users_agg_new.limit(10).show()

# Usuarios con más número de retweets

user_retweets = tweets_sample.where("retweeted_status IS NOT null").select("retweeted_status.user.screen_name", "retweeted_status")

user_retweets = user_retweets.groupBy("screen_name")\
                 .agg({"retweeted_status": "count"})\
                 .orderBy("count(retweeted_status)", ascending=False)\
                .withColumnRenamed('count(retweeted_status)','retweeted')

user_retweets.show()

# Estudiamos los usuarios que han recibido más retweets, visualizando otra información adicional y calculando el ration de retweets por tweets

retweeted = users_agg_new.join(user_retweets, users_agg_new["screen_name"] == user_retweets["screen_name"])\
                         .orderBy("retweeted", ascending=False)

retweeted = retweeted.withColumn("ratio_tweet_retweeted", retweeted["retweeted"]/retweeted["tweets"])\
                    .orderBy("ratio_tweet_retweeted", ascending=False)

retweeted.limit(10).show()

# *************** VISUALIZACIÓN CON HIVE ************************

hiveContext = HiveContext(sc)

# En este caso, tenemos tablas ya guardadas en Hive.Podemos verlas con el comando:

hiveContext.tables().show()

# Cargamos la información de la tabla tweets

tweets = hiveContext.table("tweets28a")
print("\nLos datos cargados incluyen {} tweets\n".format(tweets.count()))

tweets.printSchema()

# Analizamos los tweets que están geolocalizados

hiveContext.sql('DROP TABLE IF EXISTS tweets')
hiveContext.registerDataFrameAsTable(tweets, "tweets")
tweets_place = hiveContext.sql("SELECT place.name, COUNT(text) AS tweets FROM tweets WHERE place IS NOT NULL GROUP BY place.name ORDER BY tweets DESC")
tweets_place.limit(10).show()

# Podemos hacer el mismo análisis a través de RDDs

tweets_geo = hiveContext.sql("SELECT place.name FROM tweets WHERE place IS NOT NULL")
tweets_place_rdd = tweets_geo.rdd
tweets_place = tweets_place_rdd.toDF()

tweets_place = tweets_place.withColumn("tweets", lit(1))

tweets_place = tweets_place.groupBy("name")\
                .agg({"tweets": "sum"})\
                .orderBy("sum(tweets)", ascending=False)\
                .withColumnRenamed('sum(tweets)','tweets')

tweets_place.show()

# ****************** CONTAR HASHTAGS ************************

# Cargamos todos los tweets que no son retweets

non_retweets = hiveContext.sql("SELECT text FROM tweets WHERE retweeted_status IS NULL")

# Función que comprueba si una palabra es un hashtag

def contains_hash(str):
    for i in range (0,len(str)):
        if str[i]=='#': return(True)
    return False
    
# Limpiado de texto
    
def removePunctuation(text):

    cleanUp = re.sub(r'[^a-z0-9\s#]','',text.lower().strip())
    
    def spaceRepl(matchobj):
        if matchobj.group(0) == ' ': return ' '
        else: return ' '

    cleanUp_ns = re.sub(' {1,10000}', spaceRepl, cleanUp) 
    
    return(cleanUp_ns)

def cleanUpEnters(text):
    CleanUp = re.sub('\n',' ',text.lower().strip())
    return (CleanUp)

hashtags = non_retweets.rdd.flatMap(lambda line: line.text.split(' ')).map(lambda x: cleanUpEnters(x)).map(lambda x: removePunctuation(x)).flatMap(lambda line: line.split(' ')).filter(lambda x: contains_hash(x)==True).map(lambda x: (x,1)).reduceByKey(lambda x, y: x + y)

print ("\Hashtags con más ocurrencias:\n")

hashtagsTable = hashtags.toDF().withColumnRenamed('_1','hashtag')\
                .withColumnRenamed('_2','num')\
                .orderBy("num", ascending=False)\

hashtagsTable.limit(20).show()


# ******************* SAMPLING DE LOS DATOS **********************

# Si tenemos una base de datos muy grandes, podemos hacer un analísis de una muestra extraida

seed = 21 # Para hacer la extracción aleatoria del 1% de datos
fraction = 0.01

tweets_sample = tweets.sample(fraction, seed)

print("Número de tweets muestreados: {0}".format(tweets_sample.count()))

hiveContext.sql('DROP TABLE IF EXISTS tweets_sample')
hiveContext.registerDataFrameAsTable(tweets_sample, "tweets_sample")

# Estudiamos el número de tweets a cada hora del día

tweets_timestamp = hiveContext.sql("SELECT created_at FROM tweets_sample")

tweets_timestamp = tweets_timestamp.withColumn("hour", lit(hour(from_utc_timestamp(tweets_timestamp.created_at, 'GMT+1'))))\
                                    .withColumn("day", lit(date_format(tweets_timestamp.created_at,'MM-dd-YY')))

tweets_hour_day = tweets_timestamp.groupBy("hour","day")\
                        .agg({"created_at":"count"})\
                        .orderBy("count(created_at)", ascending=False)\
                        .withColumnRenamed('count(created_at)','count')
                        
print ("Número de tweets en cada hora en cada día:\n")
tweets_hour_day.limit(20).show()

tweets_hour = tweets_hour_day.drop("day")\
                .groupBy("hour")\
                .agg({"count": "sum"})\
                .withColumnRenamed('sum(count)','tweets')\
                .orderBy('tweets', ascending=False)\
                
print ("Número de tweets en cada hora (horas con mayor actividad de tweets):\n")
tweets_hour.show()

print ("Número de tweets vs hora del día\n")
x = tweets_hour.toPandas().hour.to_numpy()
y = tweets_hour.toPandas().tweets.to_numpy()

plt.bar(x, y)

# ****************** ESTRATIFICADO *******************

# Cargamos información de las provincias de España de la tabla 'province_28a' ya existente en Hive

province = hiveContext.sql("SELECT * FROM province_28a")
province.limit(20).show()

# Visualizamos para cada capital, el ratio de tweets por diputado

hiveContext.sql('DROP TABLE IF EXISTS tweets_place')
hiveContext.registerDataFrameAsTable(tweets_place, "tweets_place")

info_tweets_province = hiveContext.sql('SELECT tweets_place.name AS capital, tweets_place.tweets, province_28a.diputados, tweets_place.tweets/province_28a.diputados AS ratio_tweets_diputado FROM tweets_place INNER JOIN province_28a ON tweets_place.name = province_28a.capital ORDER BY ratio_tweets_diputado ASC')

info_tweets_province.limit(20).show()



