from gensim.models.keyedvectors import KeyedVectors
import pickle as pk
import collections
import numpy as np
import re

# load the pre-trained word vecters
w = KeyedVectors.load_word2vec_format('/home/liuyang/google_vec/GoogleNews-vectors-negative300.bin',encoding='utf-8',binary=True)
print('load vectors ok!')

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    return x_text

#CR_data
positive_dir = "./data/CR-data/positive"
negative_dir = "./data/CR-data/negative"

train_corpus = load_data_and_labels(positive_dir, negative_dir)

#build
def build(words, vector):
    corpus=[]
    for x in words:
        corpus += x.split(' ')
    count=[]
    # (word,frequency)
    count.extend(collections.Counter(corpus).most_common())
    uk_word=[]
    dict_1=dict()
    for word,_ in count:
        try:
            matrix = vector[word].reshape((1,300))
        except:
            uk_word.extend([(word,_)])
        else:
            dict_1[word]=len(dict_1)
            if len(dict_1)==1:
                w_matrix = matrix
            else:
                w_matrix = np.concatenate((w_matrix,matrix),0)
    for word,_ in uk_word:
        dict_1[word]=len(dict_1)
    uk_len = len(uk_word)+1 #random vocabulary size with '0'' index
    zero_index = len(dict_1)
    max_document_length = max([len(x.split(" ")) for x in words])
    
    corpus = [x.split(" ") for x in words]
    corpus_new = []
    for x in corpus:
        if len(x)<max_document_length:
            x = [dict_1[y] for y in x] + [zero_index]*(max_document_length - len(x))
        else:
            x = [dict_1[y] for y in x]
        corpus_new.append(x)
    return w_matrix, np.array(corpus_new), uk_len, len(dict_1)+1
w_matrix, x, uk_len, dict_length = build(train_corpus,w)

def save_input(w_matrix, x, uk_len, dict_length):
    f=open('./data/w_matrix','wb')
    pk.dump(w_matrix,f,2)
    f.close()
    print('save 1 ok!')
    print('pre_vector length: %d'%(w_matrix.shape[0]))
    f=open('./data/x','wb')
    pk.dump(x,f,2)
    f.close()
    print('save 2 ok!')
    f=open('./data/uk_len','wb')
    pk.dump(uk_len,f,2)
    f.close()
    print('save 3 ok!')
    print('unkown words length: %d'%(uk_len-1))
    f=open('./data/dict_length','wb')
    pk.dump(dict_length,f,2)
    f.close()
    print('save 4 ok!')
    print('dictionary length: %d'%(dict_length-1))
save_input(w_matrix, x, uk_len, dict_length)
