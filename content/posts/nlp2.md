+++
authors = ["Mike Harmon"]
title = "Text Classification 2: Natural Language Toolkit"
date = "2020-01-14"
tags = [
    "NLTK",
    "Scikit-Learn",
    "NLP"
]
series = ["NLP"]
aliases = ["migrate-from-jekyl"]
+++

## Table of Contents
----------------------

__[1. Where We Left Off](#first-bullet)__

__[2. NLTK: Stop Words, Stemming & Lemmatization](#second-bullet)__

__[3. Hyperparameter Tunning With GridSearchCV](#third-bullet)__

__[4. Conclusion](#fourth-bullet)__


---------

## Where We Left Off <a class="anchor" id="first-bullet"></a>
-----------

In the last [blogpost](http://michael-harmon.com/posts/nlp1) we covered text classification using <a href="http://scikit-learn.org/">Scikit-learn</a> and [Imbalance-Learn](https://imbalanced-learn.readthedocs.io/en/stable/) on summaries of papers from [arxiv](https://arxiv.org). We went over the basics of term frequency-inverse document frequency, Naive Bayes and Support Vector Machines. We additionally discussed techniques for handling imbalanced data both the data level and the algorithm level. In this post we'll pick up where we left off and cover uses of the [Natural Language Toolkit (NLTK)](https://www.nltk.org/) and hyperparameter tunning. Specifically we will discuss stop words, stemming and lemmatization on the previously mentioned classifiers.

First thing we need to do is connect to our [MongoDB](https://www.mongodb.com/) database:


```python
import pymongo
conn = pymongo.MongoClient('mongodb://mongodb:27017')
db   = conn.db_arxiv
```

Then get the data in the Pandas dataframe format again:


```python
import pandas as pd

# projection for subselecting only `text` and `category` fields
project = {"_id":0,"text":1,"category":1}

# get the training set
train_df = pd.DataFrame(db.train_cs_papers.find({},project))

# get the testing set
test_df = pd.DataFrame(db.test_cs_papers.find({},project))
```

Let's relabel our target variable. We create the mapping between the target and text category as well as the `y_test` vector for one vs rest classification to make the ROC and precission/recall curve:


```python
from sklearn.preprocessing import LabelEncoder, label_binarize
import numpy as np

labeler            = LabelEncoder()
train_df["target"] = labeler.fit_transform(train_df["category"])
test_df["target"]  = labeler.transform(test_df["category"])

# get the number of classes
n_classes = len(train_df["target"].unique())

# classes = [0,1,2,3]
classes   = np.sort(train_df["target"].unique())

# relabel the test set
y_test = label_binarize(test_df["target"], 
                        classes=classes)

mapping = dict(zip(labeler.classes_, range(len(labeler.classes_))))
print(mapping)
```

    {'ai': 0, 'cv': 1, 'ml': 2, 'ro': 3}


Let's remind ourselves of where we left off with the modeling by looking at the weighted Support Vector Classifier we left off with:


```python
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfTransformer)

from sklearn.metrics        import balanced_accuracy_score
from sklearn.pipeline       import Pipeline 

from sklearn.svm import LinearSVC

svm_pipe = Pipeline([('vect',    CountVectorizer()),
                     ('tfidf',   TfidfTransformer()),
                     ('model',   LinearSVC(class_weight='balanced',
                                           random_state=50))])
```

We use our model [evaluator function](https://github.com/mdh266/DocumentClassificationNLP/blob/master/evaluator.py) and [partial](https://docs.python.org/2/library/functools.html) so that we only have to feed in the different pipeline each time we want to call it:


```python
from utils.evaluator import evaluate_model
from functools import partial

evaluate_pipeline = partial(evaluate_model,
                            train_df,
                            test_df,
                            mapping)

evaluate_pipeline(svm_pipe)
```

                  precision    recall  f1-score   support
    
              ai       0.88      0.91      0.89       500
              cv       0.94      0.92      0.93       500
              ml       0.87      0.89      0.88       500
              ro       0.89      0.75      0.81        75
    
        accuracy                           0.90      1575
       macro avg       0.89      0.86      0.88      1575
    weighted avg       0.90      0.90      0.90      1575
    
    
    balanced_accuracy 0.8636666666666667


The ROC and precision/recall curves for this model are,


```python
from utils.Plot_ROC_PR_Curve import plot_roc_pr

y_pred = svm_pipe.decision_function(test_df["text"])

plot_roc_pr(y_pred = y_pred, y_test = y_test)
```


    
![png](nlp2_files/nlp2_11_0.png)
    


Now lets improve our models using the Natural Language Toolkit!

## NLTK: Stop Words, Stemming, & Lemmatization <a class="anchor" id="second-bullet"></a>
------------

### Stop Words
We can look to improve our model by removing <a href="https://en.wikipedia.org/wiki/Stop_words">stop words</a>, which are common words in the english language and do not add any information into the text. These includes words such as, "the", "at", "is", etc.  Let's look at an example using the Natural Language Toolkit ([NLTK](https://www.nltk.org/#)).  First we get an example document that we can show the effect of what removing stop words from a document does.


```python
# example document
doc = train_df["text"][242]
print(doc)
```

    E-RES is a system that implements the Language E, a logic for reasoning about
    narratives of action occurrences and observations. E's semantics is
    model-theoretic, but this implementation is based on a sound and complete
    reformulation of E in terms of argumentation, and uses general computational
    techniques of argumentation frameworks. The system derives sceptical
    non-monotonic consequences of a given reformulated theory which exactly
    correspond to consequences entailed by E's model-theory. The computation relies
    on a complimentary ability of the system to derive credulous non-monotonic
    consequences together with a set of supporting assumptions which is sufficient
    for the (credulous) conclusion to hold. E-RES allows theories to contain
    general action laws, statements about action occurrences, observations and
    statements of ramifications (or universal laws). It is able to derive
    consequences both forward and backward in time. This paper gives a short
    overview of the theoretical basis of E-RES and illustrates its use on a variety
    of examples. Currently, E-RES is being extended so that the system can be used
    for planning.


We import the nltk package and download the data required for stopwords.


```python
import nltk

for package in ['stopwords','punkt','wordnet']:
    nltk.download(package)
    
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
```

    [nltk_data] Downloading package stopwords to /home/jovyan/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.
    [nltk_data] Downloading package punkt to /home/jovyan/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.
    [nltk_data] Downloading package wordnet to /home/jovyan/nltk_data...
    [nltk_data]   Unzipping corpora/wordnet.zip.


Now we collect the stop words as a set called `stop_words`. To see the impact of removing stop words we tokenize the example document above, filter it for stop words, and use the join method to make it a string again:


```python
# collect the stopwords
stop_words    = set(stopwords.words('english')) 

# tokenize the words 
tokens  = word_tokenize(doc.replace("\n", " "))

# remove stop words from each line/list
import re 
pattern          = re.compile('[\W_]+',re.UNICODE)
filtered_tokens  = filter(lambda x : len(x) > 1, (pattern.sub("",token).lower() 
                                                  for token in tokens if token not in stop_words))

print(" ".join((filtered_tokens)))
```

    eres system implements language logic reasoning narratives action occurrences observations semantics modeltheoretic implementation based sound complete reformulation terms argumentation uses general computational techniques argumentation frameworks the system derives sceptical nonmonotonic consequences given reformulated theory exactly correspond consequences entailed modeltheory the computation relies complimentary ability system derive credulous nonmonotonic consequences together set supporting assumptions sufficient credulous conclusion hold eres allows theories contain general action laws statements action occurrences observations statements ramifications universal laws it able derive consequences forward backward time this paper gives short overview theoretical basis eres illustrates use variety examples currently eres extended system used planning


You may we removed stop words and punctuation as well as converting the characters to lowercase.  

The [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) class also has the ability to remove stop words by declaring to remove them in the constructor.  We could use this approach, but instead lets' create our own tokenizer that removes stop words so that we can add stopwords outside of those predefined by Scikit-Learn if needed:


```python
class StopWordTokenizer(object):
    """
    StopWordsTokenizer tokenizes words and removes stopwords that are 
    passed in through the the constructor.
    """
    def __init__(self, stop_words):
        import re
        self.stop_words = stop_words
        self.pattern    = re.compile('[\W_]+',re.UNICODE)

    def __call__(self, doc):
        unfiltered_tokens = (self.pattern.sub("",token) for token in word_tokenize(doc.replace("\n", " ")) 
                             if token not in stop_words)
        
        return list(filter(lambda x : len(x) > 1, unfiltered_tokens))
```

Let's now see the impact this has on our SVC model:


```python
svm_pipe2  = Pipeline([('vect',    CountVectorizer(tokenizer=StopWordTokenizer(stop_words))),
                       ('tfidf',   TfidfTransformer()),
                       ('model',   LinearSVC(class_weight='balanced',
                                            random_state=50))])

evaluate_pipeline(svm_pipe2)
```

                  precision    recall  f1-score   support
    
              ai       0.88      0.90      0.89       500
              cv       0.94      0.92      0.93       500
              ml       0.86      0.88      0.87       500
              ro       0.90      0.75      0.82        75
    
        accuracy                           0.89      1575
       macro avg       0.90      0.86      0.88      1575
    weighted avg       0.89      0.89      0.89      1575
    
    
    balanced_accuracy 0.8616666666666667


An improvement in the precision of Robotics, but an over slight decline in the balanced accuracy.  


Let's visualize the TF-IDF matrix and the most token/words as we did in the first [post](http://michael-harmon.com/blog/NLP1.html):


```python
from utils.feature_plots import plot_tfidf

plot_tfidf(pipe    = svm_pipe2,
           labeler = labeler,
           X       = train_df["text"],
           y       = train_df["target"],
           vect    = "vect",
           tfidf   = "tfidf",
           top_n   = 25)
```

<img src="https://github.com/mdh266/TextClassificationApp/blob/master/notebooks/images/tfidf.png?raw=1">

Comparing the above results to the previous [post](http://michael-harmon.com/blog/NLP1.html) we see the most important words are no longer "of" and "the", but much more sensible words like "search",  "image", "learning", and "robot".  

We can notice though that there are multiple words that really refer to the same thing, for example in the Robotics articles, "robot", "robotics", and "robots" are really refering to the same thing. If we can reduce these words to the common root word "robot" we can reduce the dimensionality and hopefully the sparisity in dataset. Doing this should help our model performance as [high dimensional problems and sparsity in your dataset can cause issues](https://stats.stackexchange.com/questions/274720/why-is-it-a-big-problem-to-have-sparsity-issues-in-natural-language-processing). In the next section we'll discuss strategies to reduce the dimensions in our dataset.

### Stemming & Lemmatization

Now let's try using [Stemming and Lemmaitization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) to improve the model performance. Stemming and Lemmatization are two processes that reduce words down to a simplier form, i.e. their "root".  This reduces the variations in words and hence the dimensionality in our model. You can see some of my work with Stemming [here](http://michael-harmon.com/blog/SentimentAnalysisP2.html). Stemming is rather rudimentary and only looks at and acts on individual words, reducing them to the simplier form.  Lemmatization on the otherhand depends on correctly identifying the intended part of speech and meaning of a word in a sentence, as well as within the larger context surrounding that sentence, such as neighboring sentences or even an entire document. I should note that stemming is known to [improve recall and degrade precision](https://en.wikipedia.org/wiki/Lemmatisation).

We use the [Snowball Stemmer](https://www.nltk.org/_modules/nltk/stem/snowball.html) and [WordNetLemmatizer](https://www.nltk.org/_modules/nltk/stem/wordnet.html) from the NLTK and show what it does to the previous example document:


```python
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

lemmer  = WordNetLemmatizer()
stemmer = SnowballStemmer(language='english')

stemmed_tokens   = (pattern.sub("",stemmer.stem(token)) 
                    for token in word_tokenize(doc.replace("\n", " ")) 
                    if token not in stop_words)

stemmed_tokens   = filter(lambda x : len(x) > 1, stemmed_tokens)

lemmatized_tokens   = (pattern.sub("",lemmer.lemmatize(token))
                      for token in word_tokenize(doc.replace("\n", " ")) 
                      if token not in stop_words)

lemmatized_tokens = filter(lambda x : len(x) > 1, lemmatized_tokens)

print("StopWords + Stemming:\n")
print(" ".join(stemmed_tokens))
print()
print("StopWords + Lemmatization:\n")
print(" ".join(lemmatized_tokens))
```

    StopWords + Stemming:
    
    er system implement languag logic reason narrat action occurr observ semant modeltheoret implement base sound complet reformul term argument use general comput techniqu argument framework the system deriv sceptic nonmonoton consequ given reformul theori exact correspond consequ entail modeltheori the comput reli complimentari abil system deriv credul nonmonoton consequ togeth set support assumpt suffici credul conclus hold er allow theori contain general action law statement action occurr observ statement ramif univers law it abl deriv consequ forward backward time this paper give short overview theoret basi er illustr use varieti exampl current er extend system use plan
    
    StopWords + Lemmatization:
    
    ERES system implement Language logic reasoning narrative action occurrence observation semantics modeltheoretic implementation based sound complete reformulation term argumentation us general computational technique argumentation framework The system derives sceptical nonmonotonic consequence given reformulated theory exactly correspond consequence entailed modeltheory The computation relies complimentary ability system derive credulous nonmonotonic consequence together set supporting assumption sufficient credulous conclusion hold ERES allows theory contain general action law statement action occurrence observation statement ramification universal law It able derive consequence forward backward time This paper give short overview theoretical basis ERES illustrates use variety example Currently ERES extended system used planning


We can see that stemming is very aggressive in reducing words to their root form while lemmatization does not blindly change words to their roots.  It turns out as only "captures" and "properties" were changed.  Let's now use both in our model by modifying the `StopWordTokenizer` from before:


```python
class StemTokenizer(object):
    """
    StemTokenizer tokenizes words, removes stopwords and stems words
    in each document.
    """
    def __init__(self, stop_words):
        import re
        from nltk.stem import SnowballStemmer
        self.stop_words = stop_words
        self.stemmer    = SnowballStemmer(language='english')
        self.pattern    = re.compile('[\W_]+',re.UNICODE)
        
    def __call__(self, doc):
        unfiltered_tokens = (self.pattern.sub("",self.stemmer.stem(token))  
                             for token in word_tokenize(doc.replace("\n", " ")) 
                             if token not in self.stop_words)
        
        return list(filter(lambda x : len(x) > 1, unfiltered_tokens))

class LemmaTokenizer(object):
    """
    LemmaTokenizer tokenizes words, removes stopwords and lemmatizes words
    in each document.
    """
    def __init__(self, stop_words):
        import re
        from nltk.stem import WordNetLemmatizer
        
        self.stop_words = stop_words
        self.lemmatizer = WordNetLemmatizer()
        self.pattern    = re.compile('[\W_]+',re.UNICODE)
        
    def __call__(self, doc):
        unfiltered_tokens = (self.pattern.sub("",self.lemmatizer.lemmatize(token))  
                             for token in word_tokenize(doc.replace("\n", " ")) 
                             if token not in self.stop_words)
        
        return list(filter(lambda x : len(x) > 1, unfiltered_tokens))
```

Note that we **first remove stop words and then stem/lemmatize words.** Now let's see how these effect the model performance. Instead of iteratively going through them and seeing which one performs the best, well just perform a grid search and take the best model.  We'll go over the details of the [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) more in the next section, but we'll use it here to evaluate the performance of each of the Tokenizer classes:


```python
from sklearn.model_selection import GridSearchCV

params = {'vect__tokenizer': (StopWordTokenizer(stop_words=stop_words),
                              StemTokenizer(stop_words=stop_words),
                              LemmaTokenizer(stop_words=stop_words))}

# 5 fold cross validation
svm_grid_search = GridSearchCV(estimator  = svm_pipe, 
                               param_grid = params, 
                               scoring    = "balanced_accuracy",
                               cv         = 5,
                               n_jobs     =-1)

# fit the models
svm_gs_model = svm_grid_search.fit(train_df["text"], 
                                   train_df["target"])
```

We can then see which pre-processing routine performed best:


```python
print(svm_gs_model.best_estimator_.steps[0][1])
```

    CountVectorizer(analyzer='word', binary=False, decode_error='strict',
                    dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
                    lowercase=True, max_df=1.0, max_features=None, min_df=1,
                    ngram_range=(1, 1), preprocessor=None, stop_words=None,
                    strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
                    tokenizer=<__main__.StemTokenizer object at 0x7f680ce231d0>,
                    vocabulary=None)


The best model in the grid search used the Stemmer. Now let's get the pefromance on the test set:


```python
svm_pipe3  = Pipeline([('vect',   CountVectorizer(tokenizer=StemTokenizer(stop_words))),
                      ('tfidf',   TfidfTransformer()),
                      ('model',   LinearSVC(class_weight='balanced',
                                            random_state=50))])

evaluate_pipeline(svm_pipe3)
```

                  precision    recall  f1-score   support
    
              ai       0.89      0.89      0.89       500
              cv       0.93      0.92      0.93       500
              ml       0.86      0.89      0.87       500
              ro       0.88      0.77      0.82        75
    
        accuracy                           0.89      1575
       macro avg       0.89      0.87      0.88      1575
    weighted avg       0.89      0.89      0.89      1575
    
    
    balanced_accuracy 0.8678333333333333


An improvement over all! However, the precision in Robotics went down and the recall went up which is a [known pheonema](https://stackoverflow.com/questions/10369479/does-stemming-harm-precision-in-text-classification#:~:text=By%20stemming%20a%20user%2Dentered,expense%20of%20reducing%20the%20precision.). Let's take a look what Stemming did to the TF-IDF Matrix:


```python
from utils.feature_plots import plot_tfidf

plot_tfidf(pipe    = svm_pipe3,
           labeler = labeler,
           X       = train_df["text"],
           y       = train_df["target"],
           vect    = "vect",
           tfidf   = "tfidf",
           top_n   = 25)
```

<img src="https://github.com/mdh266/TextClassificationApp/blob/master/notebooks/images/tfidf2.png?raw=1">

We can see that in Robotics we have reduced the terms "robot", "robotics", and "robots" to "robot" as we wished!  However, stemming is aggressive and we can see words like "image" have been redued to "imag" which can be slightly harder to interpret.

Next we'll look at hyperparameter tunning to see if we can improve the model performance further.

## HyperParameter Tuning With GridSearchCV <a class="anchor" id="third-bullet"></a>
----------------

Not only do [Scikit-llearn pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) allow us to swap out our model much easier, (say replace our Support Vector Classifier with a another model like Logistic Regression), but they also allow us to assemble sequential operations that can be evaulated together through cross-validated while choosing different parameters. 
    
To try different values of the hyperparametrs, pipelines enable setting parameters of the various steps using the pipeline stage name and the parameter name separated by a ‘__’. Notice how when we wish to change the model parameter `C` (regularization constant) by including on "model" and not the `model` object.

We can perform the grid search with 5-fold cross validation in parallel by setting `cv = 5` and `n_jobs=-1`.  We use our scoring metric as `balanced_accuracy` to account for the imbalanced classes when doing the grid search.  This is another the way we tune our algorithm for handling imbalanced data. **We should note that GridSearchCV will automatically use KFold stratified cross validation when using `cv = N` where `N` is an integer.**


```python
from sklearn.model_selection import GridSearchCV

svm_params = {'vect__min_df'     : (1,5),
              'model__loss'      : ('hinge', 'squared_hinge'),
              'model__C'         : (1, 0.5, 0.25, 0.1)}

svm_grid_search = GridSearchCV(estimator   = svm_pipe3, 
                               param_grid  = svm_params, 
                               scoring     = "balanced_accuracy",
                               cv          = 5,
                               n_jobs      =-1)

svm_gs_model = svm_grid_search.fit(train_df["text"], 
                                   train_df["target"])
```

We can look at the best estimator from this grid search:


```python
print(svm_gs_model.best_estimator_)
```

    Pipeline(steps=[('vect',
                     CountVectorizer(tokenizer=<__main__.StemTokenizer object at 0x7f576b6f6cd0>)),
                    ('tfidf', TfidfTransformer()),
                    ('model',
                     LinearSVC(C=0.1, class_weight='balanced', loss='hinge',
                               random_state=50))])


We can persist the best Support Vector Classifier model to disk using [joblib](https://scikit-learn.org/stable/modules/model_persistence.html) as it can be more efficient than pickle:


```python
import joblib 
joblib.dump(svm_gs_model.best_estimator_, '../models/weighted_svm.joblib') 
```

We can then load the model again and use it to get the model performance on the test set:


```python
from sklearn.metrics import (classification_report,
                             balanced_accuracy_score)

model  = joblib.load('../models/weighted_svm.joblib') 
pred   = model.predict(test_df["text"])
y_pred = model.decision_function(test_df["text"])

print(classification_report(test_df["target"],
                            pred, 
                            target_names=mapping))

print("\nbalanced_accuracy", balanced_accuracy_score(test_df["target"], 
                                                     pred))

plot_roc_pr(y_pred = y_pred, y_test = y_test)
```

                  precision    recall  f1-score   support
    
              ai       0.90      0.92      0.91       500
              cv       0.93      0.90      0.92       500
              ml       0.86      0.89      0.87       500
              ro       0.87      0.73      0.80        75
    
        accuracy                           0.90      1575
       macro avg       0.89      0.86      0.87      1575
    weighted avg       0.90      0.90      0.90      1575
    
    
    balanced_accuracy 0.8613333333333334



    
![png](nlp2_files/nlp2_46_1.png)
    



```python
type(test_df["text"][0])
```




    str




```python
import joblib

model  = joblib.load('../models/weighted_svm.joblib') 
```

We should note that the best esimator from the grid search has a lower test score than that of `svm_pipe3`, however, since it the hyperparameters of `svm_pipe3` are included in the gridsearch we can assume the best esimator has lower variance than `svm_pipe3`.

We can then look at the effect grid search has on the feature importance on the Robotics class using the plots introduced in the [prior post](http://michael-harmon.com/blog/NLP1.html):


```python
from utils.feature_plots import plot_coefficients

plot_coefficients(
    pipe       = svm_pipe3,
    tf_name    = 'vect',
    model_name = 'model',
    ovr_num    = mapping["ro"],
    title      = "Witout Grid Search",
    top_n      = 10
)


plot_coefficients(
    pipe       = model,
    tf_name    = 'vect',
    model_name = 'model',
    ovr_num    = mapping["ro"],
    title      = "With Grid Search",
    top_n      = 10
)
```

<img src="https://github.com/mdh266/TextClassificationApp/blob/master/notebooks/images/coeff1.png?raw=1">

<img src="https://github.com/mdh266/TextClassificationApp/blob/master/notebooks/images/coeff2.png?raw=1">

We can see that 'vechil' and 'motion' became much more important for predicting Robotics while 'network' and 'action' are more important to predicting not-Robitics.  


Now let's look at the [learning curve](https://en.wikipedia.org/wiki/Learning_curve_(machine_learning)) for this model to see if we have a high bias or high variance problem.  We first combine the train and test set and define a 10 fold stratified Cross Validation split:


```python
from sklearn.model_selection import learning_curve, StratifiedKFold

cv = StratifiedKFold(n_splits=10, random_state=None)

# combine the train/test sets
X = pd.concat([train_df["text"],test_df["text"]], axis=0)
y = pd.concat([train_df["target"],test_df["target"]], axis=0)
```

Then get the [learning curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html) information using Scikit-Learn and plot it using Plotly again. Note that we are using "balanced_accuracy" for scoring and stratified cross validation to deal with the imbalance in the classes. We use stratified cross validation to make sure that each of the K validation sets have the same proportion of targets as the entire training set. This is to make sure that our validation sets don't over or under represent any class compared to their representation in the entire dataset.


```python
from utils.learningcurve import plot_learning_curve

train_sizes, train_scores, test_scores = learning_curve(model, X, y,
                                                       cv=cv, 
                                                       n_jobs=-1,
                                                       scoring="balanced_accuracy",
                                                       train_sizes=np.linspace(.1, 1.0, 5))

plot_learning_curve(train_sizes  = train_sizes,
                    train_scores = train_scores,
                    test_scores  = test_scores)
```

<img src="https://github.com/mdh266/TextClassificationApp/blob/master/notebooks/images/learningcurve.png?raw=1">

It's a little weird that the test set accuracy grows so quickly, but overall we can see the test set accuracy and training set accuracy converge to a little over 90% which isn't bad. I would think that initially starting out with a corpus of size 2,000 there is high variance in the model as the number of features is over 10x larger than the number of samples. We can see this again in the relatively large sampling error of the test set. This leads me to believe the problem is high variance and we should look to reduce the variance in our model.

I also suspect there there is still some bias in the model that we need to address as this is a multi-class classification problem with imbalanced classes, but in general to improve the model performance we could,

1. Get more data (especially if we can balance the classes)
2. Look at the documents where there are missifications to understand what words are causing issues.
3. Investigate methods to reduce bias or variance in model such as dimensionality reduction or trying different models.


Improving model performance will have to wait for another post though!

## Conclusion <a class="anchor" id="fourth-bullet"></a>

In this blogpost we picked up from the [last one](http://michael-harmon.com/blog/NLP1.html) and went over using the Natural Language Toolkit to improve the performance of our text classification models. Specifically, we went over how to remove stopwords, stemming and lemmitization. We applied each of these to the weighted Support Vector Machine model and performed a grid search to find the optimal parameters to use for our models. One thing I would improve in the future is the preprocessing speed, it took quite a while to remove stop words and stem the text and definitely left room for improvement.

In the next post we'll work on creating a REST API from this model and using the REST API from a web app for predictions. In subsequent posts well look at ways to reduce the dimensionality of the problem so that we can use a model that is faster to train than the SVM.
