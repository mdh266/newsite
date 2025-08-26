+++
authors = ["Mike Harmon"]
title = "Text Classification 5: Fine Tuning BERT With HuggingFace"
date = "2025-08-01"
description = "Fine Tuning BERT With HuggingFace"
tags = [
    "LLMs",
    "Hugging Face"
]
series = ["LLMs"]
aliases = ["migrate-from-jekyl"]
+++



__[1. Introduction](#first-bullet)__

__[2. Collecting Data](#second-bullet)__

__[3. Hugging Face Datasets, Tokenizers & Models](#third-bullet)__

__[4. Fine Tuning BERT and Hugging Face Model Hub](#fourth-bullet)__

__[5. Using The Model With Hugging Face Pipelines](#fifth-bullet)__

__[6. Next Steps](#sixth-bullet)__



## 1. Introduction <a class="anchor" id="first-bullet"></a>
-----------------------------------------------------------

In this notebook, I will walk through the complete process of fine-tuning a [BERT (Bidirectional Encoder Representations from Transformers)](https://en.wikipedia.org/wiki/BERT_(language_model)) model using the [HuggingFace ecosystem](https://huggingface.co/). BERT has become a cornerstone of modern NLP due to its ability to capture bidirectional context and deliver strong performance across a wide range of language understanding tasks such as classification, named entity resolution and question answering. In this post I will build off of [prior posts on text classification](https://michael-harmon.com/blog/NLP4.html) by fine tuning a BERT model to classify the topic of papers in [arxiv](arxiv.org) by their abstract text. By the end of this post, I will have a working, fine-tuned BERT model ready for inference on the [Hugging Face Model Hub](https://huggingface.co/models).

The first thing is I'll be using [Google Colab](https://colab.research.google.com/) to get access to a free [CUDA](https://developer.nvidia.com/cuda-toolkit) enabled GPU. On that platform I needed install the [arxiv](https://pypi.org/project/arxiv/) and [evaluate](https://huggingface.co/docs/evaluate/en/index) libraries since they are not pre-installed:


```python
# !pip install arxiv
# !pip install evaluate
```

 Next I authenticate myself as my Google account user. This will be helpful since I will be storing the documents as json in [Google Cloud Storage](https://cloud.google.com/storage?hl=en). Authentication through [Colab](https://colab.research.google.com/) means there's no extra steps or API keys needed for me to access the data!


```python
import google.colab as colab
colab.auth.authenticate_user()
```

Now I can get started with collecting the data!

Last note that I'll make is that all the output cells have been copied to markdown cells as Colab was giving me issues with rendering the notebook on GitHub.

## 2. Collecting The Data <a class="anchor" id="second-bullet"></a>
--------------------------------------------------------------------

In [prior posts](https://michael-harmon.com/blog/NLP1.html) I obtained documents for classification by collecting paper abstracts from [arxiv](https://arxiv.org/). I was going to reuse those same documents for subsequent posts, but over the years I lost them. :( So, instead I'll use the [arixv package](https://lukasschwab.me/arxiv.py/arxiv.html) to create a new dataset for classification. I will use 3 classes or topics for the papers which I chose to be 'Artificial Intelligence', 'Information Retrieval' and 'Robotics'.

First I collect 1,000 papers on 'Ariticial Intelligence', 1,000 papers on 'Information Retrieval' and 100 on 'Robotics' using a function I wrote called [get_data](utils.py).


```python
from utils import get_arxiv_data

df = get_arxiv_data()
```


```python
df.head(2)
```

|    | id                                | code   | text                                                                            |
|---:|:----------------------------------|:-------|:--------------------------------------------------------------------------------|
|  0 | http://arxiv.org/abs/cs/9308101v1 | cs.AI  | Because of their occasional need to return to shallow points in a search ...       |
|  1 | http://arxiv.org/abs/cs/9308102v1 | cs.AI  | Market price systems constitute a well-understood class of mechanisms that ...      |

In the above results the `id` is the url of the paper, the `code` is the class label and `text` is the abstract of the paper.

I want to be able to predict the category of the abstract based of the text. This means we need to convert the category into a numerical value. [Scikit-learn's LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) is the tool for the job,


```python
from sklearn.preprocessing import LabelEncoder

labeler  = LabelEncoder()
df = df.assign(label=labeler.fit_transform(df["code"]))
```

Now each text has an associated numerical value in the column `label` with values based on the `code` value,


```python
df.head(2)
```

|    | id                                | code   | text                                                                            |   label |
|---:|:----------------------------------|:-------|:--------------------------------------------------------------------------------|--------:|
|  0 | http://arxiv.org/abs/cs/9308101v1 | cs.AI  | Because of their occasional need to return to shallow points in a search ...        |       0 |
|  1 | http://arxiv.org/abs/cs/9308102v1 | cs.AI  | Market price systems constitute a well-understood class of mechanisms that ...     |       0 |

The numerical value for each code is given by the order in the `classes_` attribute of the labler. This means mapping between the code (for the paper topic) and the label can be found by the following,


```python
{v:k for k,v in enumerate(labeler.classes_)}
```

```
{'cs.AI': 0, 'cs.IR': 1, 'cs.RO': 2}
```

Next I need to break the datasets into train, validation and test sets. I can do this with [Scikit-Learn's train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
                                            df["text"],
                                            df["label"],
                                            test_size=0.15,
                                            random_state=42,
                                            stratify=df["label"])

X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=0.20,
                                                  random_state=42,
                                                  stratify=y_train)
```

The size of the datsets are,


```python
X_train.shape, X_val.shape, X_test.shape
```

```
((1428,), (357,), (315,))
```

These are small datasets, but luckily using fine tuning I can still build a high performance model! I know that Scikit-Learn uses stratified sampling by default, but I am going check to make sure the distribution of class labels is consistent between the train, validation and test sets.


```python
from utils import plot_target_distribution_combined
plot_target_distribution_combined(y_train, y_val, y_test)
```


<figure>
<img src="https://github.com/mdh266/FineTuning/blob/main/images/distribution.png?raw=1" alt="PERF" width="700" height="500" class="center">
</figure>

You can see that it distribution of classes across each dataset is consistent.

The last thing to do before modeling is combine `X` and `y` back into one dataframe and save them to [Google Cloud Storage](https://cloud.google.com/storage?hl=en). This is necessary so I can come back to this project over time and still work with the same data.


```python
import pandas as pd
from datasets import Dataset

# train
(Dataset.from_pandas(
              pd.DataFrame({"text": X_train, "label": y_train}),
              preserve_index=False)
        .save_to_disk("gs://harmon-arxiv/train_abstracts")
)

# validation
(Dataset.from_pandas(
              pd.DataFrame({"text": X_val, "label": y_val}),
              preserve_index=False)
        .save_to_disk("gs://harmon-arxiv/val_abstracts")
)

# test
(Dataset.from_pandas(
              pd.DataFrame({"text": X_test, "label": y_test}),
              preserve_index=False)
        .save_to_disk("gs://harmon-arxiv/test_abstracts")
)
```

## 3. HuggingFace Datasets, Tokenizers & Models <a class="anchor" id="third-bullet"></a>
-------------------------------------------------------------------------------------------

Now that I have the data in [Google Cloud Storage](https://cloud.google.com/storage?hl=en) I can begin the fine tuning of my model. Since this is a classification problem I'll use a [Encoder model](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)); specifically a Bidirectional Encoder Representations from Transformers [BERT](https://huggingface.co/docs/transformers/en/model_doc/bert) model. BERT's architecture is pictured below,

<figure>
<img src="https://github.com/mdh266/FineTuning/blob/main/images/bert.png?raw=1" alt="BERT" width="300" height="500" class="center">
<figcaption>From https://towardsdatascience.com/a-complete-guide-to-bert-with-code-9f87602e4a11/
</figure>

I won't go over much about Encoders or Transformers as the internet has plently of excellent material. I found [Andrew Ng's Sequence Models](https://www.coursera.org/learn/nlp-sequence-models/paidmedia?specialization=deep-learning) course along with the [100 Page Large Language Models Book](https://www.thelmbook.com/) very helpful in understanding transformers.

This post will focus on how to fine tune a BERT model for text classification using the [Hugging Face API](https://huggingface.co/). I have heard of Hugging Face for years, but never fully understood what it is. I am currently making my way through the [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) and figured I would solidify my learnings by writing this post. Hugging Face is an open-soure platform and api for building and sharing artificial intelligence models (as well as datasets to build them). It is frequently called the "Git Hub" of AI models. With the Hugging Face API you can very easily download a pre-trained model, fine tune it for your problem and the push it back to their "Model Hub" where others in the community can use it. And I'll be doing just that in this post! The last thing I'll say about Hugging Face is that the Python library works as a high level wrapper around deep learning frameworks such as [PyTorch](https://pytorch.org/) (which I'll use), [TensorFlow](https://www.tensorflow.org/) and [JAX](https://docs.jax.dev/en/latest/).

The first thing I do is import Pandas (to reload the data from cloud storage) as well as the necessary [PyTorch](https://pytorch.org/) and Hugging Face modules.


```python
# PyTorch imports
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Hugging Face imports
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset, DatasetDict, load_from_disk
import evaluate
```

Now I can load the datasets from cloud storage the `load_from_disk` from the [Hugging Face Datasets](https://huggingface.co/docs/datasets/en/index) library


```python
train_dataset = load_from_disk("gs://harmon-arxiv/train_abstracts")
val_dataset = load_from_disk("gs://harmon-arxiv/val_abstracts")
test_dataset = load_from_disk("gs://harmon-arxiv/test_abstracts")
```

Then combine them into a [DatasetDict](https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/main_classes#datasets.DatasetDict) obect. This is not necessary, but it is convenient since applying a transformation to the DatasetDict applies it all the Datasets. This avoids repeating the same transformations across each dataset individually.


```python
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})
```

Next I download the [BERT](https://huggingface.co/docs/transformers/en/model_doc/bert) model from [HuggingFace's Model Hub](https://huggingface.co/models) as well as its associated [Tokenizer](https://huggingface.co/docs/transformers/en/main_classes/tokenizer). To do so, I use the [AutoTokenizer and AutoModelForSequenceClassification classes](https://huggingface.co/docs/transformers/en/model_doc/auto) as they allow me to swap out models easily. Notice that the tokenizer has to match the model and we have to use the [from_pretrained class methods](https://www.geeksforgeeks.org/python/classmethod-in-python/) for each class. This ensures that the tokenizer and weights for the model are both initialized from the same point in pre-training.


Lastly, notice I move the model to the GPU and that I have to put the number of classes in `AutoModelForSequenceClassification` during instantiation. Addint the number of classes adds a linear layer with softmax on top of the foundational BERT model.


```python
checkpoint = "google-bert/bert-base-uncased"
device="cuda"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
model = model.to(device)
```

One thing I will call out is that the tokenizer here is not a word level tokenization like I have used in [prior blog posts](https://michael-harmon.com/blog/NLP1.html) that used the [bag of words](https://en.wikipedia.org/wiki/Bag-of-words_model) model. Instead BERT uses a [sub-word tokenization method](https://huggingface.co/learn/llm-course/chapter6/6?fw=pt). The [100 Page Large Language Models Book](https://www.thelmbook.com/) had a good explanation on this topic, albiet it focused on [Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/en/chapter6/5?fw=pt) while BERT uses a [WordPiece tokenization](https://huggingface.co/learn/llm-course/en/chapter6/6).

I can see that the model I have downloaded is a BERT model by looking at its type:

```
type(model)
```

it returns,

```
transformers.models.bert.modeling_bert.BertForSequenceClassification
```

and
```
print(model)
```

which will return,


```
BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=768, out_features=3, bias=True)
)
```

The "classifier" layer (aka the "classification head") is the linear that was added to the model when I downloaded it. The `out_features` parameter that shows the output has 3 classes.

Now I can tokenize the datsets by creating a `tokenize_function` and applying it to the DataDict with the `map` method.


```python
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
```

Notice that I have the parameter `batched=True`, however, we have not used any padding. I will use [Dynamic Padding](https://huggingface.co/learn/llm-course/en/chapter3/2#dynamic-padding) which will determine the maximum length of documents per batch. The maximum length of documents will determine the amount of padding to be used at a batch level. If I did not use batching with Dynamic Padding all batches would have to have the same padding would have to be read in to determine the length of the longest document. Once this has been determined the padding size for each document can be ascertained. In my case, this is not such a big deal since the dataset is already in memory, but when reading large datasets from disk it can critical as loading the entire dataset in memory would be infeasible.

To use Dynamic Padding I use the [DataCollatorWithPadding](https://huggingface.co/docs/transformers/en/main_classes/data_collator) class:


```python
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

This will be used later on during training since it's just adding 0's at the beginning or at end of the tokenized vector (`token_ids`) within each batch. I can see the schema of the datasets by looking at the columns:

```
tokenized_datasets["test"].features

```

```
{'text': Value('string'),
 'label': Value('int64'),
 'input_ids': List(Value('int32')),
 'token_type_ids': List(Value('int8')),
 'attention_mask': List(Value('int8'))}
 ```

HuggingFace requires that the datasets only have the following columns:

* `labels`: The class for the text.

* `input_ids`: Vector of integers for the numerical representation of tokenized text.
  
* `attention_mask`: List of 0's or 1's for the model to infer if it should "attend" to this token in the attention mechanism.

In order to get the dataset to meet this requirements I will drop the "text" column and rename the "label" column to "labels",


```python
tokenized_datasets = tokenized_datasets.remove_columns("text")
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
```

Since I will be using [PyTorch](https://pytorch.org/) as a backend I have to convert the arrays in the datasets into PyTorch tensors.


```python
tokenized_datasets = tokenized_datasets.with_format("torch")
```

Lastly, I can confirm the schema,

```
print(tokenized_datasets["test"].features)
```

```
{'labels': Value('int64'),
'input_ids': List(Value('int32')),
'token_type_ids': List(Value('int8')),
'attention_mask': List(Value('int8'))}
```
and the size of the datasets
```
print(tokenized_datasets.num_rows)
```

```
{'train': 1428, 'validation': 357, 'test': 315}
```

## 4. Fine Tuning BERT and Hugging Face Model Hub <a class="anchor" id="fourth-bullet"></a>
-------------------------------------------------------------------------------------------

Now I can finally turn to fine tuning the model to classify arxiv papers as either "Artificial Intelligence", "Informationl Retrieval" or "Robotics." Fine tuning is process of fixing the weights in deeper layers of the Encoder, but updating the weights of the classification head as well as some shallow layers. Fine tuning will make use of the patterns learned in during pre-training in the foundational model and use them to predict the topics of the documents.

After fine tuning the model I'll upload it to the model hub. So the first thing I need to do is log in to Hugging Face Hub,


```python
from huggingface_hub import notebook_login
notebook_login()
```

Next, I chose the multiclass [ROC-AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) metric to measure the performance of the model. This is a pretty standard metric for classification problems since it in essence measures "how well the model call separate the classes." Though it should be noted the ROC-AUC curve can be misleading when you have imbalanced classes as I discussed in a [prior post](https://michael-harmon.com/blog/NLP1.html).

In order to use metrics to evaluate the performance of Hugging Face models users must use the [evaluate](https://huggingface.co/docs/evaluate/en/index) library from Hugging Face. I use the [one vs. rest multi-class ROC-AUC](https://huggingface.co/spaces/evaluate-metric/roc_auc). In order to pass it into the Hugging Face fine tunning library I have to define the following function:


```python
from typing import Tuple
import numpy as np

def compute_metrics(eval_preds):
    roc_auc_score = evaluate.load("roc_auc", "multiclass")
    preds, labels = eval_preds
    scores = torch.nn.functional.softmax(torch.tensor(preds), dim=-1)

    return roc_auc_score.compute(prediction_scores=scores, references=labels, multi_class="ovr")
```

Since I'll be pushing the model to the [Hugging Face Model Hub](https://huggingface.co/models) I'll need to create a repo and I can do it by going to my profile and clicking on the `+ New Model` tab. I'll see the new model repo form shown below:


<figure>
<img src="https://github.com/mdh266/FineTuning/blob/main/images/repo.png?raw=1" alt="REPO" width="900" height="650" class="center">
</figure>

Now that I have created the repo, I'll need to create the model. During the fine tuning process I'll update versions of the model to the Model Hub and need to specity where to push the results. I also need to define the training parameters of fine tuning. I'll do all this in the `TrainingAgruments` object below


```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
    hub_model_id="mdh266/arxivist",
    report_to="none"
)
```

These parameters train the model with 16 examples per batch from the training dataset and evaluate it with 8 examples per batch from the validation dataset. It checkpoints models both to the the [Hugging Face Model Hub](https://huggingface.co/models) (`push_to_hub=True`, specifically to the repo `hub_model_id="mdh266/arxivist"`) as well as to the local dicetory `output_dir=./results`. The checkpointing occurs at the end of each epoch (`save_strategy="epoch"`) when the model is evaluated (`eval_strategy="epoch"`). I'll point out  that the last parameter `report_to="none"` turned off the auto logging to [Weights and Biases](https://wandb.ai/site/), for some reason this occurred on Colab, but not on my laptop.

Next the trainer needs to be defined which includes the model, tokenizer, training agruments object datasets, metrics to compute and the data collator for dynamic padding.


```python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

```

Then we can begin the fine tuning process with the command below!


```python
output = trainer.train()
```

The results are below,

<figure>
<img src="https://github.com/mdh266/FineTuning/blob/main/images/performances.png?raw=1" alt="PERF" width="500" height="300" class="center">
</figure>

Finally there is one last push to the Model Hub I need to do. This push will upload all the metadata associated with fine tuning and create a basic [Hugging Face model card](https://huggingface.co/docs/hub/en/model-cards).


```python
trainer.push_to_hub("mdh266/arxivist")
```

Now the model will predict text classes 0,1,2, however, in order to get the model to predict the class names "Artificial Intelligence", "Information Retrieval" and "Robotics" the model object needs to be modified and uploaded individually. So I will grab the model,


```python
model = trainer.model
```

In order to get the class labels I need to add the mappings between the labels and the class numbers to the model configuration:


```python
model.config.label2id = {v:k for k,v in enumerate(['Artificial Intelligence','Information Retrieval', 'Robotics'])}
model.config.id2label = {k:v for k,v in enumerate(['Artificial Intelligence','Information Retrieval', 'Robotics'])}
# push to model hub
model.push_to_hub("mdh266/arxivist")
```

I'll also upload the tokenizer as well:


```python
tokenizer = trainer.processing_class
tokenizer.push_to_hub("mdh266/arxivist")
```

## 5. Using the model With Hugging Face Pipelines <a class="anchor" id="fifth-bullet"></a>
-------------------------------------------------------------------------------------------


Now I can test the model out by downloading from Model Hub using the [Hugging Face Pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines) class that bundles the tokenizer, model and post processing (to map the class numbers to class labels). This will allow end users to go from text to model class label.


```python
from transformers import pipeline

classifier = pipeline("text-classification", model="mdh266/arxivist")
```

Now I'll grab abstracts from the [arxiv.org](https://arxiv.org/) to test with the model I created.


```python
# https://arxiv.org/abs/2508.06296
# artificial intelligence
with open("../texts/ai.txt", "r") as f:
    text = f.read()

classifier(text)
```

```[{'label': 'Artificial Intelligence', 'score': 0.979738712310791}]```


```python
# https://arxiv.org/abs/2508.05633
# information retrieval
with open("../texts/ir.txt", "r") as f:
    text = f.read()

classifier(text)
```

```[{'label': 'Information Retrieval', 'score': 0.9323310852050781}]```

The pipeline class gives the class label (`label`) as well the probability the model gave that prediction (`score`).

I can even do predictions on the test set by converting it to a list of texts:



```python
classifier(test_df["text"].sample(2).to_list())
```

```
[{'label': 'Robotics', 'score': 0.9148617386817932},
 {'label': 'Information Retrieval', 'score': 0.9640209674835205}]
 ```

To get the ROC-AUC score on the test set I need to write a function that will calculate it using the [DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) class from PyTorch to enable dynamic padding. The function is below,


```python
from typing import Dict
import numpy as np
from torch.utils.data import DataLoader

def calculate_roc_auc(model, loader: DataLoader) -> Dict[str, np.float64]:

  roc_auc_score = evaluate.load("roc_auc", "multiclass")
  model.eval()
  for batch in loader:
      batch = {k: v.to(device) for k, v in batch.items()}
      with torch.no_grad():
          outputs = model(**batch)
          scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
          roc_auc_score.add_batch(references=batch["labels"],
                                prediction_scores=scores)

  return roc_auc_score.compute(multi_class="ovr")
```

Since I need the actual class probabilities for all classes for ROC-AUC then I cannot use the pipeline classifier directly, but instead I must get the model directly,


```python
model = classifier.model
```

Then I can just use the `DataLoader` class as before and pass it to the above function.


```python
testset_dataloader = DataLoader(
    tokenized_datasets["test"], batch_size=8, collate_fn=data_collator
)

calculate_roc_auc(model, testset_dataloader)
```

```
{'roc_auc': np.float64(0.9821414141414141)}
```

A pretty good ROC-AUC!!

## 6. Next Steps <a class="anchor" id="sixth-bullet"></a>
----------------------------------------------------------
In this notebook, I successfully fine-tuned a BERT model using the HuggingFace transformers library and achieved strong performance, with a ROC-AUC score of approximately 0.98 on the test set. This demonstrates BERT’s ability to generalize well when trained with high-quality data and an appropriate fine-tuning strategy.

We covered the full pipeline from dataset preparation to model evaluation and showcased how to use the model for inference. This approach can be easily adapted to a variety of other NLP tasks such as sentiment analysis. One thing I will explore in the future is adding weights to a custom loss function through PyTorch to help deal with the imbalance of classes in the dataset.


