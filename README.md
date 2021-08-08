
<img align="right" width="300" height="" src="https://upload.wikimedia.org/wikipedia/commons/1/1e/Logo-Goethe-University-Frankfurt-am-Main.svg">

# New state-of-the-Art in Preposition Sense Disambiguation

**Supervisor:**
* [Prof. Dr. Alexander Mehler](https://www.texttechnologylab.org/team/alexander-mehler/)
* [Alexander Henlein](https://www.texttechnologylab.org/team/alexander-henlein/)


**Institutions:**
  * **[Goethe University](http://www.informatik.uni-frankfurt.de/index.php/en/)**
  * **[TTLab - Text Technology Lab](https://www.texttechnologylab.org/)**
  


## Project Description

The disambiguation of words is a central part of NLP tasks. In particular, there is the ambiguity of prepositions,
which has been a problem in NLP for over a decade and still is.
For example the preposition 'in' can have a temporal (e.g. in 2021) or a spatial (e.g. in Frankuft) meaning.
A strong motivation behind the learning of these meanings are current research attempts to transfer text to artifical scenes. 
A good understanding of the real meaning of prepositions is crucial in order for the machine to create matching scenes.


With the birth of the transformer models in 2017 [[1]](#1), attention based models have been pushing boundries in many NLP disciplines. 
In particular, [bert](https://huggingface.co/transformers/model_doc/bert.html), a transformer model by google and pre-trained on more than 3,000 M words,
obtained state-of-the-art results on many NLP tasks and Corpus.

The goal of this project is to use modern transformer models to tackle the problem of preposition sense disambiguation.
Therefore, we trained a simple bert model on the [SemEval 2007 dataset](https://www.clres.com/elec_dictionaries.php#tppcorp) [[2]](#2), a central benchmark dataset for this task.
To the best of our knowledge, the best purposed model for disambiguating the meanings of prepositions on the SemEval achives an accuracy of up to 88% [[3]](#3).
Neither more recent approaches surpass this frontier[[4]](#4)[[5]](#5) .
Our model achives an accuracy of 90.84%, out-performing the current state-of-the-art. 

## How to train

To meet our goals, we cleand the SemEval 2007 dataset to only contain the needed information. 
We have added it to the repository and can be found in `./data/training-data.tsv`.

**Train a bert model:** </br>
First, install the requirements.txt. Afterwards, you can train the bert-model by:

``` python3 trainer.py --batch-size 16 --learning-rate 1e-4 --epochs 4 --data-path "./data/training_data.tsv" ```

The chosen hyper-parameters in the above example are tuned and already set by default. 
After training, this will save the weights and config to a new folder `./model_save/`.
Feel free to omit this training-step and use [our trained weights directly](https://drive.google.com/drive/folders/1LGSeQt7TK-p4Lq_inJiBo4BMPL27c0Yq).


## Examples

We attach an example tagger, which can be used in an interactive manner.
```python3 -i tagger.py```

Sourrond the preposition for which you like to know the meaning in `<head>...</head>` and feed it to the tagger:

```
>>> tagger.tag("I am <head>in</head> big trouble")
Predicted Meaning: Indicating a state/condition/form, often a mental/emotional one that is being experienced 

>>> tagger.tag("I am speaking <head>in</head> portuguese.")
Predicted Meaning: Indicating the language, medium, or means of encoding (e.g., spoke in German)

>>> tagger.tag("He is swimming <head>with</head> his hands.")
Predicted Meaning: Indicating the means or material used to perform an action or acting as the complement of similar participle adjectives (e.g., crammed with, coated with, covered with)

>>> tagger.tag("She blinked <head>with</head> confusion.")
Predicted Meaning: Because of / due to (the physical/mental presence of) (e.g., boiling with anger, shining with dew)
```



## References

<a id="1">[1]</a> 
Vaswani, Ashish et al. (2017). 
Attention is all you need. 
Advances in neural information processing systems. P. 5998--6008.

<a id="2">[2]</a> 
Litkowski, Kenneth C and Hargraves, Orin (2007). 
SemEval-2007 Task 06: Word-sense disambiguation of prepositions.
Proceedings of the Fourth International Workshop on Semantic Evaluations (SemEval-2007). P. 24--29

<a id="3">[3]</a> 
Litkowski, Ken. (2013). 
Preposition disambiguation: Still a problem. 
CL Research, Damascus, MD.

<a id="4">[4]</a> 
Gonen, Hila and Goldberg, Yoav. (2016). 
Semi supervised preposition-sense disambiguation using multilingual data. 
Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers. P. 2718--2729

<a id="5">[5]</a> 
Gong, Hongyu and Mu, Jiaqi and Bhat, Suma and Viswanath, Pramod (2018). 
Preposition Sense Disambiguation and Representation.
Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. P. 1510--1521



