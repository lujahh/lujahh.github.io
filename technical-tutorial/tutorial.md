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

## Following along? 
You can follow this tutorial with our git repository that contains the above corpora and a docker image with all WordKit requirements. Make sure you have `docker` installed on your machine
1. Clone [this repository](https://github.com/lujahh/WordKitTutorial)
2. On the terminal, navigate to the root of the cloned directory, and run the following command:
```
docker run -it  -v "$PWD:/app/" wordkit-tutorial
```
You will be able to run all of the following code from the terminal.

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
french = lexique('corpora/Lexique382.txt', fields = ('orthography', 'phonology'))

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

Now let's read in our Spanish corpus. There's no built-in for DIMEx100, but we can still use WordKit's `reader` function to read in our file as a pandas dataframe. `reader` has the same specification as `lexique`:
```python
reader(pathtofile, fields)
```

We want to read in all of the fields we have (orthography and phonology information), so we don't need to call fields. We additionally call `sep = '\t'` because our file is tab-separated and the default for `reader` is comma-separated.

```python
spanish = reader('corpora/T22.full.dic', sep = '\t')

#there are no headers in the dataset, so add those
spanish.columns = ['orthography', 'phonology']

#how many words are in DIMEx100?
print("{} words in Spanish corpus".format(len(spanish)))
#11476 words in Spanish corpus

#what does our dataframe look like? 
print(spanish[100:110])
#     orthography          phonology
# 100     ACEPTAS      a s e p t a s
# 101      ACERCA       a s e r( k a
# 102    ACERCABA     a s e r( k a b
# 103     ACERCAR    a s e r( k a r(
# 104      ACERCO       e s e r( k o
# 105       ACERO         a s e r( o
# 106   ACERQUE_7       a s e r( k e
# 107      ACERVO       a s e r( b o
# 108   ACETILENO  a s e t i l e n o
# 109     ACICATE      a s i k a t e

#make our orthography column lowercase to match the French data
spanish['orthography'] = spanish['orthography'].str.lower()

#add a language column
spanish['language'] = 'esp'

print(spanish[100:110])
#     orthography          phonology language
# 100     aceptas      a s e p t a s      esp
# 101      acerca       a s e r( k a      esp
# 102    acercaba     a s e r( k a b      esp
# 103     acercar    a s e r( k a r(      esp
# 104      acerco       e s e r( k o      esp
# 105       acero         a s e r( o      esp
# 106   acerque_7       a s e r( k e      esp
# 107      acervo       a s e r( b o      esp
# 108   acetileno  a s e t i l e n o      esp
# 109     acicate      a s i k a t e      esp
```

## 2. Norming the phonological information
As you can see, our `phonology` column looks different in our `spanish` dataframe. We need to convert the DIMEx100 transcriptions to IPA transcriptions. We can do this by modifying the code under the hood of the `lexique` function to correspond to the [DIMEx100 transcriptions](https://turing.iimas.unam.mx/~luis/DIME/DIMEx100/t22.html):

```python
DIMEX_2IPA = {'a': 'a',
              'k': 'k',
              'p': 'p',
              'l': 'l',
              'i': 'i',
              's': 's',
              'o': 'o',
              'b': 'b',
              'd': 'd',
              'e': 'e',
              'f': 'f',
              'g': 'ɡ',
              'm': 'm',
              'n': 'n',
              't': 't',
              'r(': 'ɾ',
              'r':'r',
              'tS':'ʧ',
              'Z':'ʝ',
              'n~':'ɲ',
              'u': 'u',
              'x': 'x'}

def dimex_to_ipa(syllables):
    """Convert dimex phonemes to IPA unicode format."""
    for syll in syllables:
        yield "".join([DIMEX_2IPA[syll]])


def phon_func(string):
    """Process a phonology string."""
    phon = string.split()
    phon = [segment_phonology(x) for x in dimex_to_ipa(phon)]
    return tuple(chain.from_iterable(phon))

spanish['phonology'] = spanish['phonology'].apply(phon_func)

print(spanish[100:110])
#     orthography                    phonology language
# 100     aceptas        (a, s, e, p, t, a, s)      esp
# 101      acerca           (a, s, e, ɾ, k, a)      esp
# 102    acercaba        (a, s, e, ɾ, k, a, b)      esp
# 103     acercar        (a, s, e, ɾ, k, a, ɾ)      esp
# 104      acerco           (e, s, e, ɾ, k, o)      esp
# 105       acero              (a, s, e, ɾ, o)      esp
# 106   acerque_7           (a, s, e, ɾ, k, e)      esp
# 107      acervo           (a, s, e, ɾ, b, o)      esp
# 108   acetileno  (a, s, e, t, i, l, e, n, o)      esp
# 109     acicate        (a, s, i, k, a, t, e)      esp
```

Success! Our `french` and `spanish` dataframes are in the same format.

## 3. Featurizing the phonological information

We can now concatenate our dataframes to create one dataframe to be used in a classification task. 

```python
words = pd.concat([french, spanish], sort = False).reindex()
```
**NOTE:** Before you create your feature matrix, think about what you want as your training and test data. The below assumes that you use the entire Lexique and DIMEx100 corpora as training data, but you may want to split `words` into training and test data before fitting the transformer. 

Now it's time to create our feature matrix. WordKit contains many means for featurizing words, including ngrams, one hot features, and linear features specific to psycholinguistic models. For this tutorial, let's use bigrams.

```python
#create our transformer

p = NGramTransformer(n=2, field='phonology')

#transform into features
X_p = p.fit_transform(words)

#what's the feature vector length?
p.vec_len
#1167
```

Now you can fit your labels and build your classifier!


