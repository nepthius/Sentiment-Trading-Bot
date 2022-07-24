from bs4 import BeautifulSoup
import requests
import nltk
nltk.download('punkt') #installs dependecy for nltk word tokenizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('movie_reviews')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import movie_reviews
import random

training = [(list(movie_reviews.words(fileid)), category)
for category in movie_reviews.categories()
for fileid in movie_reviews.fileids(category)]


random.shuffle(training)

print(training[1])

#stopwords to skip over
stop_words = set(stopwords.words("english"))

#stemming
ps = PorterStemmer()

#lemmatizing
lemmatizer = WordNetLemmatizer()

#saves a requests object into xml
url = 'https://www.economist.com/finance-and-economics/rss.xml'
xml = requests.get(url)

#Creates a soup object with the xml file
soup = BeautifulSoup(xml.content, 'lxml') 
#I believe this is an xml parser, but it doesn't recognize it as such for some reason

#Creates list containing all titles and a list containing all descriptions
xml_titles = soup.find_all('title')
xml_descs = soup.find_all('description')


#Looks at the text of each individual title tag and breaks it apart into a list of words
#Filters words to exclude neutral words such as "and"
for title in xml_titles:
    t_words = word_tokenize(title.text)
    fwords = []
    for word in t_words:
        if word not in stop_words:
            fwords.append(lemmatizer.lemmatize(word))
    #print(fwords)
#Used NLTK to split words instead of split so that words like "wasn't" are split into "was" and "n't" and other utilities


#Looks at the text of each individual description tag and breaks it apart into a list of words
for desc in xml_descs:
    d_words = desc.text.split()
    #print(d_words)


#look into lemmatizers

#