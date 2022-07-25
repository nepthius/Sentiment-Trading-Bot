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
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
import io
import random
import pickle

class VoteClassifier(ClassifierI):
  def __init__(self, *classifiers):
    self._classifiers = classifiers
  
  def classify(self, features):
    votes = []
    for c in self._classifiers:
      v = c.classify(features)
      votes.append(v)
    return mode(votes)
  def confidence(self, features):
    votes = []
    for c in self._classifiers:
      v = c.classify(features)
      votes.append(v)
    cvotes = votes.count(mode(votes))
    return cvotes/len(votes)

short_pos = io.open("./Downloads/positive_reviews.txt", encoding='latin-1')
short_neg = io.open("./Downloads/negative_reviews.txt", encoding='latin-1')
short_pos = short_pos.read()
short_neg = short_neg.read()


documents = []

for r in short_pos.split('\n'):
  documents.append((r, "pos"))
for r in short_neg.split('\n'):
  documents.append((r, "neg"))

all_words = []
short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
  all_words.append(w.lower())
for w in short_neg_words:
  all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

def find_features(document):
  words = word_tokenize(document)
  features = {}
  for w in word_features:
    features[w] = (w in words)

  return features
featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)

training = featuresets[:10000]
testing = featuresets[10000:]

MNB = SklearnClassifier(MultinomialNB())
temp = open("MNB.pickle", "rb")
MNB = pickle.load(temp)
temp.close()


Bernoulli = SklearnClassifier(BernoulliNB())
temp = open("Bernoulli.pickle", "rb")
Bernoulli = pickle.load(temp)
temp.close()

LR = SklearnClassifier(LogisticRegression())
temp = open("LR.pickle", "rb")
LR = pickle.load(temp)
temp.close()

LinearSVC = SklearnClassifier(LinearSVC())
temp = open("LinearSVC.pickle", "rb")
LinearSVC = pickle.load(temp)
temp.close()



classifiers = [MNB, Bernoulli, LR, LinearSVC]

#trains classifier and prints accuracy
def print_acc(classifier, testing):
    #classifier.train(training)
    print("Accuracy: ", (nltk.classify.accuracy(classifier, testing)) *100)



for x in classifiers:
  print_acc(x, testing)


vc = VoteClassifier(MNB, Bernoulli, LR, LinearSVC)

print("voted accuracy percent: ", (nltk.classify.accuracy(vc, testing)) * 100)
print("Classification: ", vc.classify(testing[1][0]), "Confidence: ", vc.confidence(testing[1][0]))
print("Classification: ", vc.classify(testing[0][0]), "Confidence: ", vc.confidence(testing[0][0]))
print("Classification: ", vc.classify(testing[2][0]), "Confidence: ", vc.confidence(testing[2][0]))
print("Classification: ", vc.classify(testing[3][0]), "Confidence: ", vc.confidence(testing[3][0]))
print("Classification: ", vc.classify(testing[4][0]), "Confidence: ", vc.confidence(testing[4][0]))

def sentiment_classify(text):
    feats = find_features(text)
    return vc.classify(feats), vc.confidence(feats) 

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

database = {"universal":0}

#Looks at the text of each individual title tag and breaks it apart into a list of words
#Filters words to exclude neutral words such as "and"


for title in xml_titles:
    fwords = ""
    t_words = word_tokenize(title.text)
    
    #print(title.text)
    text = nltk.word_tokenize(title.text)
    #print(text)
    
    for word in text:
        #if word not in stop_words:
        fwords += " " + str((word.lower()))
    print(fwords)
    print(sentiment_classify(fwords))
    if vc.classify == "neg":
        database["universal"]-=1
    else:
        database['universal']+=1

    #print(fwords)
#Used NLTK to split words instead of split so that words like "wasn't" are split into "was" and "n't" and other utilities


#Looks at the text of each individual description tag and breaks it apart into a list of words
'''
for desc in xml_descs:
    d_words = desc.text.split()
'''
    #print(d_words)


