import os
import numpy as np
import sklearn
import scipy
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
import random

from modules.lm import BERT
from datasets import load_dataset

import random
from copy import deepcopy
from tqdm import tqdm

random.seed(42)

# ---------------------------------------------------------

def cosim(a, b):
    if (norm(a)*norm(b)) == 0: return 0
    return dot(a, b)/(norm(a)*norm(b))

def top_k_accuracy(y_hat, y_true, k = 5, THRESHOLD = 0.95):
    matched = 0
    for i in range(y_hat.shape[0]):
        # For each predicted vector => compare true ones
        similarities = np.sort(
            np.array([cosim(y_hat[i], y_true[j]) for j in range(y_true.shape[0])])
        )[::-1]
        
        has_top_k_match = int(np.sum((similarities[:k] >= THRESHOLD).astype(int)) >= 1)
        matched += has_top_k_match

    return matched / y_hat.shape[0]


def cross_validate_model(X, Y):

    # Shuffle together
    temp = list(zip(X, Y))
    random.shuffle(temp)
    X, Y = zip(*temp)
    X, Y = np.array(X), np.array(Y)

    # Cross validation
    crossVal = KFold(n_splits=10)
    K = 30
    accuracies = []
    progress = tqdm(range(crossVal.get_n_splits(X)))

    for i, (train_index, test_index) in (enumerate(crossVal.split(X))):

        model = make_pipeline(StandardScaler(), PLSRegression(n_components=K))
        model.fit(X[train_index], Y[train_index])

        # Given a predicted vector, we rank all the 534 vectors in the gold standard data set by decreasing cosine similarity values
        y_hat = model.predict(X[test_index]) 
        accuracies.append(
            top_k_accuracy(y_hat, Y)
        )
        progress.update(1)
        
    return np.mean(accuracies)


# -------------- Binder features

filename = os.path.join("data", "binder_semantic_features", "word_ratings", "WordSet1_Ratings.csv")

binder_features = {}
with open(filename, "r") as file:
    for line in file.readlines()[1:]:
        if ",na," not in line: # NOTE: incomplete data issue
            word = line.split(",")[1]
            features = [float(x) if x != "na" else 0 for x in line.split(",")[5:69]]
            binder_features[word] = np.array(features)

# -------------- BERT

def get_context_sentences(word, dataset, max_words = 1000, progress = False, patience = 500):
    word_sentences = [] 
    if progress == True: progress_bar = tqdm(range(max_words))
    cnt = 0
    word = f" {word} "
    for row in dataset:
        if len(word_sentences) >= max_words: break
        if cnt >= patience: break

        # Check wikipedia entry containing word with probability 30% (random sampling)
        if word in row["text"] and random.uniform(0,1) <= 0.3:
            cnt = 0
            text = row["text"].replace("\n", " ")
            # Pick all sentences containing that word
            for sentence in text.split("."):
                if len(word_sentences) >= max_words: break
                if word in sentence:
                    word_sentences.append(sentence)
                    if progress == True: progress_bar.update(1)
        else:
            cnt += 1
    return word_sentences

lm = BERT(device="cuda")
wikipedia = load_dataset("wikipedia", "20220301.en")

dataset = wikipedia["train"]
contexts = {}
progress_bar = tqdm(range(len(binder_features.keys())))
batch_size = 32

X = []
Y = []

for word in binder_features.keys():
    sentences = get_context_sentences(word, dataset, patience=2000)
    if len(sentences) > 1:

        # 1k context word embedding averaged
        word_embedding = np.array([lm.get_contextualized_word_embedding(word, sent) for sent in sentences]).mean(axis=0)
        """
        # Batch encoding of 1000 word context sentences
        batches = len(sentences) // batch_size + 1
        sents_embeds = []

        for i in range(batches):
            lm.get_contextualized_word_embedding(key, key)
            batch_sents = sentences[i*batch_size:(i+1)*batch_size]
            if len(batch_sents) > 0: sents_embeds.append(get_text_BERT_embed(batch_sents))
        
        # Stack and average the contexts
        sents_embeds = np.concatenate(sents_embeds)
        word_embedding = np.mean(sents_embeds, axis=0) # mean over 1k context sents embeddings
        """
        X.append(word_embedding)
        Y.append(binder_features[word])
        
    progress_bar.update(1)

X = np.array(X).astype(np.float32)
Y = np.array(Y).astype(np.float32)