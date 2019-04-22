SENT_START = "--BOS--"
SENT_END = "--EOS--"
NEWLINE = "--newline--"

#list of features to extract
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