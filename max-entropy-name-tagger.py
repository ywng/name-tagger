import argparse
import sys
import pickle, datetime
import pandas as pd
import numpy as np
import spacy
from spacy.tokens import Doc
#from nltk.classify.maxent import MaxentClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec
import nltk
from nltk.corpus import treebank
from sklearn.cluster import MiniBatchKMeans

from score import score
from const import FEAT_EXTRACT_LIST, SENT_START, SENT_END, NEWLINE

MODEL_SAVE_PATH = "./model/model_%s.sav"
VECTORIZER_SAVE_PATH = "./data/vectorizer.pickle"
REGRESSER_MAX_ITER = 2000

nltk.download('treebank')

nlp = spacy.load('en')
wv_dims = 300
word_vec = Word2Vec(treebank.sents(), size=wv_dims).wv

K = 1024
kmeans = MiniBatchKMeans(n_clusters=K, random_state=0).fit(word_vec[word_vec.vocab])

class FeatureBuilder:
  def __init__(self, sent, sent_features):
    doc = Doc(
      nlp.vocab,
      words=sent,
    )
    nlp.tagger(doc)
    nlp.parser(doc)

    self.features = sent_features

    #extract token features we desired
    for feat in FEAT_EXTRACT_LIST:
      for token in doc:
        #getattr(self, feat) will return the function
        #the function will store the corresponding feature in self.features of the token tagged by spaCy
        getattr(self, feat)(token)

    #context features
    for token in doc:
      for feat in FEAT_EXTRACT_LIST:
        # No context features for word vector
        if feat.startswith(("word_vector_cluster", "binarization", "word_vector_naive")):
          continue

        # Token -1
        if token.i == 0:
            self.features[token.i][feat + "_-1"] = SENT_START
        else:
            self.features[token.i][feat + "_-1"] = self.features[token.i - 1][feat]

        # Token - 2
        if (token.i == 0) or (token.i == 1):
            self.features[token.i][feat + "_-2"] = SENT_START
        else:
            self.features[token.i][feat + "_-2"] = self.features[token.i - 2][feat]

        # Token + 1
        if token.i == len(sent) - 1:
            self.features[token.i][feat + "_+1"] = SENT_END
        else:
            self.features[token.i][feat + "_+1"] = self.features[token.i + 1][feat]
        
        # Token + 2
        if (token.i == len(sent) - 2) or (token.i == len(sent) - 1):
            self.features[token.i][feat + "_+2"] = SENT_END
        else:
            self.features[token.i][feat + "_+2"] = self.features[token.i + 2][feat]

  def is_alpha(self, token):
      self.features[token.i]["is_alpha"] = token.is_alpha

  def is_digit(self, token):
      self.features[token.i]["is_digit"] = token.is_digit

  def is_punct(self, token):
      self.features[token.i]["is_punct"] = token.is_punct

  def is_title(self, token):
      self.features[token.i]["is_title"] = token.is_title

  def is_lower(self, token):
      self.features[token.i]["is_lower"] = token.is_lower

  def is_upper(self, token):
      self.features[token.i]["is_upper"] = token.is_upper

  def orth_(self, token):
      self.features[token.i]["orth_"] = token.orth_

  def lemma_(self, token):
      self.features[token.i]["lemma_"] = token.lemma_

  def lower_(self, token):
      self.features[token.i]["lower_"] = token.lower_

  def norm_(self, token):
      self.features[token.i]["norm_"] = token.norm_

  def shape_(self, token):
      self.features[token.i]["shape_"] = token.shape_

  def prefix_(self, token):
      self.features[token.i]["prefix_"] = token.prefix_

  def suffix_(self, token):
      self.features[token.i]["suffix_"] = token.suffix_

  def word_vector_naive(self, token):
    if token.lower_ in word_vec.vocab:
      vec = word_vec[token.lower_]
    else:
      vec = [0] * wv_dims

    for i in range(wv_dims):
      self.features[token.i]["wv_dim"+str(i)] = vec[i]

  def binarization(self, token):
    if token.lower_ in word_vec.vocab:
      vec = word_vec[token.lower_]
      pos_mean = vec[vec>0].mean()
      neg_mean = vec[vec<0].mean()
    
      '''
        U+ is a string feature which turns on when the value (Cij ) falls into 
        the upper part of the positive list. Similarly, B− refers to the bottom 
        part of the negative list. The insight behind is that we only consider 
        the features with strong opinions (i.e., positive or negative) on each 
        dimension and omit the values close to zero.
      '''
      for i in range(wv_dims):
        if vec[i]>=pos_mean:
          self.features[token.i]["wv_bin"+str(i)] = "U+"
        elif vec[i]<=neg_mean:
          self.features[token.i]["wv_bin"+str(i)] = "B-"
        else:
          self.features[token.i]["wv_bin"+str(i)] = "zero"
    else: 
      #out of vocabulary,
      for i in range(wv_dims):
          self.features[token.i]["wv_bin"+str(i)] = "oov" # oov means not in the vocab

  def word_vector_cluster(self, token):
      if token.lower_ in word_vec.vocab:
        #make it string to make it discrete feature
        self.features[token.i]["word_vector_cluster"] = str(kmeans.predict([word_vec[token.lower_]])[0]) 
      else:
        self.features[token.i]["word_vector_cluster"] = "oov" # oov means not in the vocab

class MaxEntNameTagger:

  def train(self, train_data_path):
    features_df = self.__build_features(train_data_path)
    features = list(features_df)
    features.remove("tag")
   
    #features_set_labels = list(zip(features_set, labels))
    #self.classifier = MaxentClassifier.train(features_set_labels, max_iter=NUM_EPOCHES)
    '''
      It was too slow to train the classifier using MaxntClassifier without solver algorithm.
      So, using multinomial LogisticRegression is equivalent to Max-Entrophy classifier.
    '''
    vectorizer = DictVectorizer()
    trainX = vectorizer.fit_transform(features_df[features].to_dict("records"))
    trainy = features_df["tag"].values

    self.classifier = LogisticRegression(
        multi_class="multinomial",  # Using cross-entropy loss
        solver="lbfgs",            
        C=2.0,                      
        n_jobs=-1,
        warm_start=True,
        verbose=1,
        max_iter=REGRESSER_MAX_ITER,
    )

    self.classifier.fit(trainX, trainy)
    #save the vectorizer for later use
    pickle.dump(vectorizer, open(VECTORIZER_SAVE_PATH, 'wb'), protocol=2)

  def eval(self, word_seq_path, tag_ans_path, output_path):
    self.test(word_seq_path, output_path)
    score(tag_ans_path, output_path)

  def test(self, word_seq_path, output_path):
    features_df = self.__build_features(word_seq_path)
    features = list(features_df)
    features.remove("tag")

    vectorizer = pickle.load(open(VECTORIZER_SAVE_PATH, "rb"))

    X = vectorizer.transform(features_df[features].to_dict("records"))
    Y_pred = self.classifier.predict(X)

    tokens = features_df["token"].values

    # Write out the predicted name entity tags
    with open(output_path, "w") as out:
        for token, tag in zip(tokens, Y_pred):
            # Handle newlines
            if (token == NEWLINE):
                out.write("\n")
                continue
            out.write(token + "\t" + tag + "\n")

  def __build_features(self, src_data_path):
    index = 0
    sent = []
    sent_features = {}
    all_features_data_list = []
    for cnt, line in enumerate(open(src_data_path, "r")):
      if not line.split(): #end of sent
        #end of current sent, we so build the features for the current sent so far
        builder = FeatureBuilder(sent, sent_features)
        sent_features = builder.features
        for index, token in enumerate(sent):
          columns = sorted(sent_features[index].keys())
          all_features_data_list.append([sent_features[index][c] for c in columns])

        #features of the end of sent
        eos_features = [NEWLINE] * len(columns)
        all_features_data_list.append(eos_features)
        
        index = 0
        sent = []
        sent_features = {}
          
      else:
        parts = line.strip().split("\t")
        if len(parts) == 4:
          token, pos, chunk, tag = parts
        else:
          token, pos, chunk = parts
          tag = None

        #existing features as given in the source data
        sent_features[index] = {
          "token": token,
          "pos": pos,
          "chunk": chunk,
          "tag": tag
        }
        index += 1
        sent.append(token)

      if cnt % 10000 == 0:
        print("#lines of features built: {:>8}".format(cnt))

      #if cnt == 10000:
      #  break

    df = pd.DataFrame(all_features_data_list, columns=columns)
    df.to_csv(src_data_path + "-features", index=False)

    return df

def save_model(model):
  pickle.dump(model, open(MODEL_SAVE_PATH%datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'wb'), protocol=2)

def load_model(path_to_model):
  model = pickle.load(open(path_to_model, 'rb'))
  return model

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train', dest='train', action='store_true')
  parser.add_argument('--eval', dest='eval', action='store_true')
  parser.add_argument('--test', dest='test', action='store_true')

  parser.add_argument('--train_data', type=str, help='training data file location.')

  parser.add_argument('--model', type=str, help='the path to trained model.')
  parser.add_argument('--words', type=str, help='the word sequence to tag named entity.')
  parser.add_argument('--name_tags', type=str, help='the ground true answers for the word sequence.')

  parser.add_argument('--output', type=str, help='the output path for the tagged word sequence.')


  args = parser.parse_args()

  tagger = MaxEntNameTagger()

  if args.train:
    #train the model
    if not args.train_data:
      sys.exit("Please provide the training data file path!")
    tagger.train(args.train_data)
    save_model(tagger)
  elif args.eval:
    #eval the model perf on dev corpus
    if not args.words or not args.name_tags:
      sys.exit("Please provide both the word sequence to eval and the name tag answers!")
    if not args.model:
      sys.exit("Please provide the path to trained model")
    if not args.output:
      sys.exit("Please provide the output path for storing the tagged word sequence!")
    tagger = load_model(args.model)
    tagger.eval(args.words, args.name_tags, args.output)
  elif args.test:
    #test the model on unseen text
    if not args.words:
      sys.exit("Please provide the word sequence to test!")
    if not args.model:
      sys.exit("Please provide the path to trained model")
    if not args.output:
      sys.exit("Please provide the output path for storing the tagged word sequence!")
    tagger = load_model(args.model)
    tagger.test(args.words, args.output)

if __name__ == "__main__":
  main()