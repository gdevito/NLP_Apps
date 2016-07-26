from __future__ import division
import matplotlib.pyplot as plt
import nltk
%matplotlib inline
docids = nltk.corpus.reuters.fileids()
seen_words = set()
num_tokens_vs_wordtypes = []
ntoks = 0
for docid in docids:
    tokens = nltk.corpus.reuters.words(docid)
    ntoks += len(tokens)
    for tok in tokens: seen_words.add(tok.lower())
    num_tokens_vs_wordtypes.append( (ntoks, len(seen_words)) )

plt.plot(
    [ntok for ntok,ntype in num_tokens_vs_wordtypes],
    [ntype for ntok,ntype in num_tokens_vs_wordtypes]
         )
		 
		 
from __future__ import division
from collections import defaultdict
import nltk, random

# Utility functions if you want them
# Adding simple comment to see if can push


def normalized_dict(dct):
    """
    Assume dct is a string-to-number map.  Return a normalized version where the values sum to 1.
    {"a":4.0, "b":2.0} ==> {"a":0.6666, "b":0.3333}
    """
    s = sum(dct.values())
    new_dct = {key: value*1.0/s for key,value in dct.items()}
    return new_dct

def weighted_draw_from_dict(choice_dict):
    """Randomly choose a key from a dict, where the values are the relative probability weights."""
    # http://stackoverflow.com/a/3679747/86684
    choice_items = choice_dict.items()
    total = sum(w for c, w in choice_items)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choice_items:
       if upto + w > r:
          return c
       upto += w
    assert False, "Shouldn't get here"

# Quick test: This should be approx 7500
a_count = sum(['a'==weighted_draw_from_dict({'a':3,'b':1}) for i in range(100000)])
a_count

def make_ngrams(tokens, ngram_size):
    """Return a list of ngrams, of given size, from the input list of tokens.
    Also include **START** and **END** tokens appropriately."""
    ngrams = []
    tokens = ['**START**'] * (ngram_size-1) + tokens + ['**END**'] * (ngram_size-1)
    for i in range(ngram_size, len(tokens)+1):
        ngrams.append( tuple(tokens[i-ngram_size:i]))
    return ngrams

class NgramModelCounts:
    def __init__(self):
        self.vocabulary = set()
        self.ngram_size = None
        # designed to have the structure {prefix: {nextword: count}}
        # Feel free to change if you don't like this approach
        self.ngram_counts = defaultdict(lambda:defaultdict(int))

def get_ngram_counts(sentences, ngram_size):
    """'Train' a fixed-order ngram model by doing the necessary ngram counts.
    Return a data structure that represents the counts."""
    model = NgramModelCounts()
    model.ngram_size = ngram_size
    model.vocabulary.add("**START**")
    model.vocabulary.add("**END**")
    for sent_tokens in sentences:
        ngrams = make_ngrams(sent_tokens, ngram_size)
        for ngram in ngrams:
            prefix = tuple(ngram[:ngram_size-1])
            model.ngram_counts[prefix][ngram[-1]] += 1
        for tok in sent_tokens:
            model.vocabulary.add(tok)
    return model
	
def next_word_prob(prefix, model, word_pseudocount):
    """ For the given prefix, return the distribution over all possible next words,
    using the model and per-word pseudocount value given as input. 
    Hints: 
    (0) You will want to use the normalized_dict function defined in part C1
    (1) Don't forget to add word_pseudocount to the counts in the input model. 
    (2) The input model (i.e. counts) doesn't include the **OOV** token. You will need 
        to explicitly add a counts for this (i.e. just pseudocounts).
    """
    #initialize oovs
    oovs = {"**OOV**": 0}
    
    for nextWord in model.ngram_counts[prefix].keys():
        if nextWord in model.vocabulary:
            if nextWord != "**START**":
                oovs[nextWord] = model.ngram_counts[prefix][nextWord]
        else:
            oovs["**OOV**"] += model.ngram_counts[prefix][nextWord]
    
    
    #next update pseudocounts
    for nextWord in oovs.keys():
        oovs[nextWord] += word_pseudocount
        
    return normalized_dict(oovs)
	
def sample_sentence(model):
    ##See hints above.
    ##Additional hint: don't forget to pad with START tokens. 
    tokens = []
    tokens = ['**START**'] * (model.ngram_size-1)

    count = model.ngram_size-1
    maxCount = 1000 + (model.ngram_size-1)
    tokNext = "begin"
    #allow to reach end naturally or set on maxcount
    while tokNext != "**END**":
        first = count-model.ngram_size+1
        last = count
        #set new ngram
        prefix = tuple(tokens[first:last])
        
        #get new token
        tokNext = weighted_draw_from_dict(next_word_prob(prefix, model, 0))
        tokens.append(tokNext)
        count = count + 1
        #if goes above count, manually set END
        if count >= maxCount:
            tokNext = "**END**"
            
    #finally pad sentence w/ END tokens based on ngram_size
    tokens += ["**END**"]*(model.ngram_size-1)
    return tokens


import math
def logprob(tokens, model, word_pseudocount):
    # return lp, the log probability of the tokens under the model 
    # it also prints the info described above
    
    lp = 0
    tokens = ['**START**']*(model.ngram_size-1) + tokens + ['**END**']
    for t in range(model.ngram_size-1, len(tokens)):
        start,end = t-model.ngram_size+1, t
        prefix = tuple(tokens[start:end])
        probs = next_word_prob(prefix, model, word_pseudocount)
        nextword = tokens[t]
        if nextword not in model.vocabulary:
            nextword = '**OOV**'
        prob = probs[nextword]
        print prefix, nextword, prob, "with count", model.ngram_counts.get(prefix,{}).get(nextword,0)
        lp += math.log(prob)
    return lp
