import pickle
import numpy as np

#load the cached files using pickle
loaded_embeddings=pickle.load(open('Cached_Data/loaded_embedding','rb'))
words=pickle.load(open('Cached_Data/words','rb'))
idx2words=pickle.load(open('Cached_Data/idx2words','rb'))


#suction to compute cosine similarity between 50 dimentional representation of keyword and entire word embedding
def cos_sim(b,A,epsilon = 1e-5):
    """compute consine similarity between A and b
    """
    dot_product = np.dot(A, b)
    norm_a = np.linalg.norm(A,axis=1) + epsilon
    norm_b = np.linalg.norm(b) + epsilon
    return dot_product / (norm_a * norm_b)


def find_nearest(ref_vec, words, embedding,topk=3):
    """
    Finds the top-k most similar words to "word" in terms of cosine similarity in the given embedding
    :param ref_vec: reference word vector
    :param words: dict, word to its index in the embedding
    :param embedding: numpy array of shape [V, embedding_dim]
    :param topk: number of top candidates to return
    :return a list of top-k most similar words
    """
    # compute cosine similarities
    scored_words = cos_sim(ref_vec, loaded_embeddings)
    
    # sort the words by similarity and return the topk
    sorted_words = np.argsort(-scored_words)
    
    #return thr list of similar words
    return [(idx2words[w]) for w in sorted_words[:topk]]

#print(find_nearest(loaded_embeddings[words['hate']], words, loaded_embeddings, topk=4))