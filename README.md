# Name Tagger
In this program, we are implementing a name tagger using Maximum Entropy Markov Model. We shall use the corpus from the 2003 CoNLL (Conference on Natural Language Learning), which uses texts from the Reuters news service. This corpus identifies three types of names: person, organization, and location, and a fourth category, MISC (miscellaneous) for other types of names.

The training model using NLTK MaxentClassifier is essentially feature-based supervised learning. Several features we can used/combined as features for the model learning:
1. Characteristic of the word itself: whether the word is title, contain numbers, is punctuation etc. Also, the shape (U.K is X.X), the lemma (canonical form e.g. running / run), prefix, suffix, leftmost token of this token’s syntactic descendants (left_edge) might help with the name entity classification
2. Contextual features: looking at the word itself is not enough. Whether the word is of a person/organization/location sometimes depends on neighboring words. So, I will put the word characteristc features of the left two and right one neighboring words, as contextual features.
3. External source to help: these days, word embedding is a common and helpful representation about text. We can group similar words together using word embedding. If we know a certain word's group, it might help with the classification as word embedding group gives word meaning.

Below are some of the features I tried of the above 3 main categories.
FEAT_EXTRACT_LIST = [
  "is_title",
  "orth_",
  "lemma_",
  "lower_",
  "norm_",
  "shape_",
  "prefix_",
  "suffix_",
  "is_alpha",
  "is_digit",
  "is_punct",
  "like_num",
  "is_upper",
  "dep_",
  "is_stop",
  "left_edge",
]

48254 out of 51578 tags correct
  accuracy: 93.56
5917 groups in key
3905 groups in response
3244 correct groups
  precision: 83.07
  recall:    54.83
  F1:        66.06


In this case, I tried to include as many word characteristics as possible and together with the context(-2, +2) combination of them.
It gives okay accuracy, but the F1 score is low. This shows that too much unnecessary features may worsen the performance.

***
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

49368 out of 51578 tags correct
  accuracy: 95.72
5917 groups in key
4862 groups in response
4136 correct groups
  precision: 85.07
  recall:    69.90
  F1:        76.74

Now, I have less word characteristics features, the performance is increased.

***
FEAT_EXTRACT_LIST = [
  "is_title",
  "lemma_",
  "norm_",
  "shape_",
  "suffix_",
  "dep_",
  "left_edge",
]

49293 out of 51578 tags correct
  accuracy: 95.57
5917 groups in key
4816 groups in response
4100 correct groups
  precision: 85.13
  recall:    69.29
  F1:        76.40

Switch a little bit to different word characterirstics, the performance is more or less the same. Probably, the lemma, shape of the word, and its contextual features play the important role.

***
FEAT_EXTRACT_LIST = [
  "is_title",
  "orth_",
  "lemma_",
  "lower_",
  "norm_",
  "shape_",
  "prefix_",
  "suffix_",
  "word_vector_cluster",
]

50043 out of 51578 tags correct
  accuracy: 97.02
5917 groups in key
5493 groups in response
4717 correct groups
  precision: 85.87
  recall:    79.72
  F1:        82.68

Then, I also include the word vector clustering group as a feature. Initially, with the same number of epoches of training (9 epoches), I just got slightly, negligibly improvement over the accuracy and F1 score.
Later on, I realised that the last bit of push for training accuracy (from 99.1 --> 99.8) with more epoches (19 epoches) help with the performance. Now the accuracy and F1 score is much improved.

***
FEAT_EXTRACT_LIST = [
  "is_title",
  "orth_",
  "lemma_",
  "norm_",
  "shape_",
  "suffix_",
  "word_vector_cluster",
]

50212 out of 51578 tags correct
  accuracy: 97.35
5917 groups in key
5729 groups in response
4870 correct groups
  precision: 85.01
  recall:    82.31
  F1:        83.63

Now, I try to change the total number of word vector embbedings group (K = 1024), and further increased the number of epoches (25 epoches), and just take backward 2, forward 1 neighboring word as contextual features, the performance is pushed higher a bit. F1 score increase is not trivial. Probably, the hyper-parameters (K, number of epoches, how many forward and backward neighboring words we consider) is also important.


## Setup and Libraries
I use [spaCy](https://spacy.io/) python library to help extracting text features. It support convenient features extraction about text e.g. is_upper, prefix, suffix, shape of the word, is_stop, lemma (the canonical form) of a word etc.
```
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
nltk.download()
```

## How to run
ssh into nyu mscis crunchy 1
```
ssh ywn202@access.cims.nyu.edu
ssh crunchy1.cims.nyu.edu
```

### Training
It will produce a model object in ./model/
```
python max-entropy-name-tagger.py --train --train_data ./data/CONLL_train.pos-chunk-name
```

### Eval
Running below script will tag the given word sequence. It will evalute and print the accuracy. 
For my implementation, 
accuracy: 
precision: 
recall: 
F1: 
--model: path to the trained model file.   
--words: the words file to be tagged.   
--name_tags: the named entity tags ground true.   
--output: the path where the predicted entity tag result will be output to.  
```
python max-entropy-name-tagger.py --eval --model ./model/model_2019-04-03_19-28-43.sav --words ./data/CONLL_dev.pos-chunk --name_tags ./data/CONLL_dev.name --output ./output/CONLL_dev.predicted_name
```

### Test
--model: path to the trained model file.  
--words: the words file to be tagged.  
--output: the path where the predicted entity tag result will be output to.  
```
python max-entropy-name-tagger.py --test --model ./model/model_2019-04-03_19-28-43.sav --words ./data/CONLL_test.pos-chunk --output ./output/CONLL_test.name
```