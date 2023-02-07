import argparse
import numpy as np
import json
from tqdm import tqdm

import torch
import torch.nn as nn

class Word2VecMean(nn.Module):
    def __init__(self, vocab_size, embedding_size) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, input_var):
        output = list()
        for var in tqdm(input_var):
            embedded = self.embedding(var)
            mean = embedded.mean(dim=0)
            output.append(mean.detach().numpy())

        return output

def data_load(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.readlines()

    new_sample = list()
    name = list()
    for sample in data:
        new_sample.append(sample.split("\t")[1])
        name.append(sample.split("\t")[0])

    return new_sample, name

def set_vocab(sample):
    sample_list = list()
    word_set = list()
    for data in sample:
        sample_list.append(data.split())
        word_set += data.split()
        word_set = list(set(word_set))

    word_set.sort()
    vocab = {word: i for i, word in enumerate(word_set)}

    return vocab

def get_var(sample, vocab):
    sample_list = list()
    for data in sample:
        sample_list.append(data.split())

    var = list()
    for new_data in sample_list:
        index = list()
        for word in new_data:
            index.append(vocab[word])

        var.append(torch.LongTensor(index))

    return var

def run(word2vec, var, name, file_path):
    if "all" in file_path:
        save_path = "word2vec_all"
    else:
        save_path = "word2vec"
    output = word2vec(var)

    result = np.array(output)
    np.save(save_path + "/" + save_path + "_data.npy", result)
    name_dic = dict()
    name_dic["name"] = name
    name_json = json.dumps(name_dic)
    name_file = open(save_path + "/" + save_path + "_names.json","w")
    name_file.write(name_json)
    name_file.close()

if __name__ == "__main__":
    par = argparse.ArgumentParser()
    par.add_argument("-d", "--data", default="data/test_34_data.txt", 
                     type=str, help="data path")
    par.add_argument("-ad", "--all_data", default="data/all_data.txt", 
                     type=str, help="data path")
    par.add_argument("-em", "--embedding_size", default=64,
                     type=int, help="embedding size")
    args = par.parse_args()

    sample, name = data_load(args.all_data)
    vocab = set_vocab(sample)
    print("Vocab size: {}".format(len(vocab)))
    var = get_var(sample, vocab)

    word2vec = Word2VecMean(len(vocab), args.embedding_size)

    run(word2vec, var, name, args.all_data)

    test_sample, test_name = data_load(args.data)
    test_var = get_var(test_sample, vocab)

    run(word2vec, test_var, test_name, args.data)
