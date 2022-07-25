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
Bernoulli = SklearnClassifier(BernoulliNB())
LR = SklearnClassifier(LogisticRegression())
LinearSVC = SklearnClassifier(LinearSVC())
NuSVC = SklearnClassifier(NuSVC())

classifiers = [MNB, Bernoulli, LR, LinearSVC, NuSVC]

#trains classifier and prints accuracy
def print_acc(classifier, testing):
    classifier.train(training)
    print("Accuracy: ", (nltk.classify.accuracy(classifier, testing)) *100)



for x in classifiers:
  print_acc(x, testing)

vc = VoteClassifier(MNB, Bernoulli, LR, LinearSVC, NuSVC)

print("voted accuracy percent: ", (nltk.classify.accuracy(vc, testing)) * 100)
print("Classification: ", vc.classify(testing[1][0]), "Confidence: ", vc.confidence(testing[1][0]))
print("Classification: ", vc.classify(testing[0][0]), "Confidence: ", vc.confidence(testing[0][0]))
print("Classification: ", vc.classify(testing[2][0]), "Confidence: ", vc.confidence(testing[2][0]))
print("Classification: ", vc.classify(testing[3][0]), "Confidence: ", vc.confidence(testing[3][0]))
print("Classification: ", vc.classify(testing[4][0]), "Confidence: ", vc.confidence(testing[4][0]))


save_MNB = open("MNB.pickle", "wb")
pickle.dump(MNB, save_MNB)
save_MNB.close()

save_Bernoulli = open("Bernoulli.pickle", "wb")
pickle.dump(Bernoulli, save_Bernoulli)
save_Bernoulli.close()

save_LR = open("LR.pickle", "wb")
pickle.dump(LR, save_LR)
save_LR.close()

save_LinearSVC = open("LinearSVC.pickle", "wb")
pickle.dump(LinearSVC, save_LinearSVC)
save_LinearSVC.close()

save_NuSVC = open("NuSVC.pickle", "wb")
pickle.dump(NuSVC, save_NuSVC)
save_NuSVC.close()

save_documents = open("documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

save_allwords = open("allwords.pickle", "wb")
pickle.dump(all_words, save_allwords)
save_allwords.close()

save_wordf = open("wordf.pickle", "wb")
pickle.dump(word_features, save_wordf)
save_wordf.close() 

save_featuresets = open("featuresets.pickle", "wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()
