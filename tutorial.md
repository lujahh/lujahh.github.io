# Making a feature matrix from multiple corpora with WordKit

Research dealing with crosslinguistic comparisons often requires the use of multiple corpora. Preprocessing with multiple data sources can be time consuming, because often different corpora are coded in different ways. 

### [WordKit](https://github.com/clips/wordkit) allows users to read in data and perform phonological norming with built-in functionality for some of the most popular natural language corpora.

This may be sufficient for many projects using English, German, Dutch, and French. 

### WordKit also allows users to easily read in any corpus in csv, tsv, or xlsx format. 

In this tutorial, we will show how to combine phonology data from [Lexique](http://www.lexique.org/), a corpus of French that has built-in support in WordKit, and [DIMEx100](https://turing.iimas.unam.mx/~luis/DIME/CORPUS-DIMEX.html), a corpus of Spanish that does not have built-in support. By the end of the tutorial, we will have a dataframe of French and Spanish words' orthography and phonology as well as an n-gram feature matrix of the phonology information. 

We cover three steps:
1. Reading in the corpora
2. Norming the phonological information
3. Featurizing the phonological information

Because WordKit creates feature matrices that are compatible with [scikit-learn](https://scikit-learn.org/stable/index.html), you will easily be able to take the outputs of this tutorial and implement a classifier that predicts whether a word is French or Spanish from its phonology.

*Let's get started!*

## Installing WordKit
You can install WordKit with pip on the command line using `pip install wordkit`. WordKit has the following requirements:
* ipapy
* numpy
* pandas
* reach
* nltk
* scikit-learn


## Corpora
The current (as of May 2021) version of Lexique available for download is 3.83. WordKit's built-in lexique reader reads version 3.82. To take full advantage of WordKit, we'll use version 3.82, available for download [here](https://github.com/lvaudor/tuto_texte_Marmiton/blob/master/Lexique382/Lexique_Lisez_Moi.txt). You can use version 3.83 with WordKit using the same steps we'll use for the DIMEx100 corpus. 

The DIMEx100 corpus is available for download from the [DIMEx100 site](https://turing.iimas.unam.mx/~luis/DIME/CORPUS-DIMEX.html). The zip file comes with three `diccionarios` that contain orthography and phonology information for the words in the corpus. We use `T22.full.dic` (the other two files contain more fine-grained phonological information).

## Imports

We use the following in this tutorial:

```python
import pandas as pd
from itertools import chain
from wordkit.corpora.base import segment_phonology
from wordkit.corpora import lexique, reader
from wordkit.features import NGramTransformer
```
## 1. Reading in the corpora
Let's first read in our French corpus using the WordKit built-in `lexique`. This function has the following specification:
```python
lexique(pathtofile, fields)
```
This returns a pandas dataframe with the specified fields taken from Lexique 3.82. If `phonology` is specified,  `lexique` returns a normed phonology column where the transcriptions from the Lexique corpus are converted to IPA transcription.

For this tutorial, we are interested in orthography and phonology information, so we can call:
```python
french = lexique("corpora/Lexique382.txt", fields = ("orthography", "phonology"))

#how many words are in Lexique 3.82? 
print("{} words in French corpus".format(len(french)))
#125733 words in French corpus

#what does our dataframe look like?
print(french[100:110])
#                phonology  orthography  length language
#119  (a, b, a, t, ə, m, ɑ̃)  abattements      11      fra
#120            (a, b, a, t)     abattent       8      fra
#121            (a, b, a, t)      abattes       7      fra
#122      (a, b, a, t, œ, ʁ)     abatteur       8      fra
#123      (a, b, a, t, œ, ʁ)    abatteurs       9      fra
#124         (a, b, a, t, e)      abattez       7      fra
#125      (a, b, a, t, j, e)     abattiez       8      fra
#126     (a, b, a, t, j, ɔ̃)    abattions       9      fra
#127      (a, b, a, t, i, ʁ)   abattirent      10      fra
#128         (a, b, a, t, i)      abattis       7      fra         

#we don't need the length column
french = french[['phonology', 'orthography', 'language']]
```

That was easy! We have our phonology information from the Lexique corpus in IPA transcripition format instead of the idiosyncratic Lexique format. This will allow us to combine the information from the Lexique corpus with other corpora. 

Now let's read in our Spanish corpus. There's no built-in for DIMEx100, but we can still use WordKit's `reader` function to read in our file. `reader` has the same specification as `lexique`:
```python
reader(pathtofile, fields)
```











