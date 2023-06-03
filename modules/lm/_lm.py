

from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import torch
from modules.algorithms import search_sequence_numpy
from tqdm import tqdm
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class LanguageModel:
    def __init__(self, device):
        self.device = device
        self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True,).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.model_max_length = 512
        self.model.eval()

    def get_noncontextualized_embeddings(self, word, layer=-1):
        """ Returns the embedding of a word without context """
        tokenized = self.tokenizer(word, return_tensors="pt").input_ids.to(self.device)
        layer_activations = self.model(tokenized)[2][1:][layer][0]
        word_embedding = layer_activations[1:-1].mean(dim=0).detach().cpu().numpy()
        
        del layer_activations
        del tokenized
        torch.cuda.empty_cache()
        return word_embedding

    def get_contextualized_embeddings(self, word, sentences, batch_size=32, progress=False):
        """ Batch process a set of contexts for a word to compute its embedding 
            
            input:      word                    String                      A word to contextualize
                        sentences               list(String)                Set of context sentences containing the word
                        batch_size              Int
                        progress                Bool                        Display progress bar
            output:     contextualized_words    np.array:(layers, N, 768)   Layers activations of the word contextualized in from each sentence
        """
        batches = math.ceil(len(sentences) // batch_size)
        word_tokens = self.tokenizer(word, return_tensors="pt").input_ids[0, 1:-1].detach().cpu().numpy()
        
        # layer, sentence, embed
        contextualized_words = torch.zeros((12, len(sentences), 768))
        k = [0] * 12 # counter for each layer
        if progress == True: progress_bar = tqdm(range(batches))

        # Encode context sentences batchwise
        for x in range(batches):
            tokenized = self.tokenizer(sentences[x*batch_size:(x+1)*batch_size], padding=True, return_tensors="pt").input_ids.to(self.device)
            layer_activations = self.model(tokenized)[2][1:] # obtain embeddings

            # for each layer, within each sentence, locate the word to contextualize and store it
            for i, layer in enumerate(layer_activations):
                for j, sent in enumerate(layer):
                    idx = search_sequence_numpy(tokenized[j].detach().cpu().numpy(), word_tokens)[0]
                    contextualized_words[i, k[i], :] = sent[idx:idx+len(word_tokens)].mean(dim=0).detach().cpu() # avg of wordpieces for contextualized embed
                    k[i] += 1
                    del sent
                del layer
            if progress == True: progress_bar.update(1)
            del layer_activations

        # mem cleaning!
        del word_tokens
        del batches
        del tokenized
        if progress == True: del progress_bar
        torch.cuda.empty_cache()
        return contextualized_words
    
    
    def get_dominant_meaning(self, word_embeddings, max_meanings = 20):
        """ Apply K-means clustering from a set of contextualized word embeddings, return centroid of dominant meaning
            
            input:      word_embeddings     np.array:(N, 768)   Multiple contextual N embeddings of the same word, embeded size 768
                        max_K               int                 Maximum number of clusters to evaluate  
            output:     clusters            np.array(N)         Labels assigned to each contextual embedding
                        dominant_meaning    np.array(768)       Word embedding of the centroid of the dominant meaning
        """
        # Choose best no.meanings
        max_k = None
        max_score = -1
        for k in list(range(2, min(len(word_embeddings)-1, max_meanings))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(word_embeddings)
            score = silhouette_score(word_embeddings, kmeans.labels_)
            if score > max_score: 
                max_score = score
                max_k = k

        # Cluster meanings
        labels = KMeans(n_clusters=max_k, random_state=42, n_init="auto").fit_predict(word_embeddings)

        # Pick dominant cluster
        dominant_label = None 
        max_l = -1
        for label in np.unique(labels):
            idx = np.where(labels == label)
            if len(idx[0]) > max_l:
                max_l = len(idx[0])
                dominant_label = label
        
        return labels, word_embeddings[np.where(labels == dominant_label)].mean(axis=0)

