# Name Tagger
It is implementing a name tagger using Maximum Entropy Markov Model. We shall use the corpus from the 2003 CoNLL (Conference on Natural Language Learning), which uses texts from the Reuters news service. This corpus identifies three types of names: person, organization, and location, and a fourth category, MISC (miscellaneous) for other types of names.

The training model using NLTK MaxentClassifier is essentially feature-based supervised learning. Several features we can used/combined for the model learning:
1. Characteristic of the word itself: whether the word is title, contain numbers, is punctuation etc. Also, the shape (U.K is X.X), the lemma (canonical form e.g. running / run), prefix, suffix, leftmost token of this word’s syntactic descendants (left_edge) might help with the name classification.
2. Contextual features: looking at the word itself is not enough. Whether the word is of a person/organization/location sometimes depends on neighboring words. So, I will put the word characteristic features of the left two and right one neighboring words, as contextual features.
3. External source to help: these days, word embedding is a common and helpful representation about text. We can group similar words together using word embedding. If we know a certain word's group, it might help with the classification as word embedding group gives word meaning.

Below are some of the features I tried of the above 3 main categories.  

***
Without the word embeddings, it produces below results (Assignment 6 baseline).

FEAT_EXTRACT_LIST = [
  "is_title",
  "orth_",
  "lemma_",
  "lower_",
  "norm_",
  "shape_",
  "prefix_",
  "suffix_",
]

50212 out of 51578 tags correct  
accuracy: 97.35  
5917 groups in key  
5729 groups in response  
4870 correct groups  
precision: 85.01  
recall: 82.31  
F1: 83.63  

***
With naive word embeddings by adding D (where D is the dimensions of the word vector) features to each token, the performance is decreased.

FEAT_EXTRACT_LIST = [
  "is_title",
  "orth_",
  "lemma_",
  "lower_",
  "norm_",
  "shape_",
  "prefix_",
  "suffix_",
  "word_vector_naive",
]

49961 out of 51578 tags correct  
  accuracy: 96.86  
5917 groups in key  
6432 groups in response  
5039 correct groups  
  precision: 78.34  
  recall:    85.16  
  F1:        81.61  

***
With binarization (convert to discrete valued-features) as described in (Gu et al 2014), the performance is not improved but decreased. It is probably because the trainer is not good at handling these features, and taking each dim of the vector as a feature is too much.  

FEAT_EXTRACT_LIST = [
  "is_title",
  "orth_",
  "lemma_",
  "lower_",
  "norm_",
  "shape_",
  "prefix_",
  "suffix_",
  #"word_vector_naive",
  "binarization",
]

48609 out of 51578 tags correct  
  accuracy: 94.24  
5917 groups in key  
7485 groups in response  
4804 correct groups  
  precision: 64.18  
  recall:    81.19  
  F1:        71.69  

***
Using word vector clustering, I got the best performance on dev dataset. 

FEAT_EXTRACT_LIST = [
  "is_title",
  "orth_",
  "lemma_",
  "lower_",
  "norm_",
  "shape_",
  "prefix_",
  "suffix_",
  #"word_vector_naive",
  #"binarization",
  "word_vector_cluster",
]


50570 out of 51578 tags correct  
  accuracy: **98.05**  
5917 groups in key  
5997 groups in response  
5195 correct groups  
  precision: 86.63  
  recall:    87.80  
  F1:        **87.21**  

## Setup and Libraries
I use [spaCy](https://spacy.io/) python library to help extracting text features. It support convenient features extraction about text e.g. is_upper, prefix, suffix, shape of the word, is_stop, lemma (the canonical form) of a word etc.
```
module load python-3.6
virtualenv --system-site-packages py3.6.3
source py3.6.3/bin/activate

pip install spacy
pip install pandas
```

Need to download some corpus that spaCy used.
```
python -m spacy download en
```

Need to download some corpus that nltk used.
Run the Python interpreter and type the commands:
Select download corpus package.
```
import nltk
nltk.download('treebank')
```

## How to run
ssh into nyu mscis crunchy 1
```
ssh ywn202@access.cims.nyu.edu
ssh crunchy1.cims.nyu.edu
```

### Training
It will produce a model object in ./model/
It will take quite long to train the classifier. The best train model saved in ./model/model_cluster_wv.sav
```
python max-entropy-name-tagger.py --train --train_data ./data/CONLL_train.pos-chunk-name
```

### Eval
Running below script will tag the given word sequence. It will evalute and print the accuracy. 
--model: path to the trained model file.   
--words: the words file to be tagged.   
--name_tags: the named entity tags ground true.   
--output: the path where the predicted entity tag result will be output to.  
```
python max-entropy-name-tagger.py --eval --model ./model/model_cluster_wv.sav --words ./data/CONLL_dev.pos-chunk --name_tags ./data/CONLL_dev.name --output ./output/CONLL_dev.predicted_name
```

### Test
--model: path to the trained model file.  
--words: the words file to be tagged.  
--output: the path where the predicted entity tag result will be output to.  
```
python max-entropy-name-tagger.py --test --model ./model/model_cluster_wv.sav --words ./data/CONLL_test.pos-chunk --output ./output/CONLL_test.name
```