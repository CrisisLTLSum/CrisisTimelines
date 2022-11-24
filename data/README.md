# Data:

* `train_notext.json`, `dev_notext.json`, and `test_notext.json` files contain the train, dev, and test splits of our dataset.
Each one of those json files contains a list of examples, where each example represents an annotated timeline and is stored as a dictionary 
with the following fields:
  - `batch_id`: an id that was used to indicate the annotation batch.
  - `timeline_id`: a unique timeline id.
  - `seed`: the seed tweet.
  - `tweet_ids`: list of tweet ids. The number of tweet ids equals to the number of `tweets` + 1 to account for the `seed` tweet.
  - `times`: list of tweet times. The number of times equals to the number of `tweets` + 1 to account for the `seed` tweet.
  - `relevance_scores`: list of lists, where each sublist corresponds to each tweet in `tweets`. Each sublist has 3 labels of 0s or 1s to 
  indicate if the tweet is part of the timeline or not. There are 3 labels because each timeline was annotated by 3 workers.
  - `rejection_reasons`: list of lists, where each sublist corresponds to the reasons the annotators gave for choosing a label (0 or 1). 
  repetitive means the tweet has repetitive information and it gets a 0 label, informative means that the tweet is not informative and it gets a 
  0 label, relevant means that the tweet it not relevant tweet and it gets a 0 label. 
  'NA' means that the tweet should be part of the timeline and it's label should be 1. There are 3 reasons for each tweet because each timeline was annotated 
  by 3 workers.
  - `majority_reasons`: a list containing the annotation reason among the 3 annotators for each tweet.
  - `rejection_agreements`: a list of Yes/No to indicate the agreement among the workers when it came to providing the annotation reason.
  - `tweets`: list of tweets.
  - `labels`: list of 0s and 1s and those represent the majority vote based on the annotation in `relevance_scores`. 
  These are the labels that we use to train and evaluate our systems on the timeline extraction task.
  - `category`: the domain of the timeline (i.e., traffic, wildfire, fire, storm, etc.).
  - `Summary 1`: summary of the first worker.
  - `Summary 2`: summary of the second worker.
  - `Summary 3`: summary of the third worker.
  
  
