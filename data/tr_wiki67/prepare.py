"""
python prepare.py train_tokenizer
python prepare.py --task=nsp_mlm --block_size=255


python prepare.py --block_size=255
"""

import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer, sent_tokenize

from datasets import Dataset, load_from_disk
import pandas as pd
import re
import os
import random
import sys


import torch
from tokenizers import Tokenizer



raw_table_path = "raw_dataset"
ab_table_path = "ab_table"
nsp_mlm_table_path = "nsp_mlm_table"
tokenizer_path = "../../wordpiece_tokenizer"



def __delete_subtitles(example):
    """
    will be used as a map function
    take every example-doc and delete if it has subtitles (if line count of word is < 5)
    """
    temp = []

    # this tokenizer will be used as to find words
    word_tokenizer = RegexpTokenizer(r'\w+')

    # split lines
    for line in example.splitlines(True):
        # if count of word is > 5 just keep this line, if not > 5, we will consider it as subtitle and not gonna keep it
        if len(word_tokenizer.tokenize(line)) > 5:
            temp.append(line)
    
    # in some cases there will be no lines that are bigger than 5
    # also some examples may not have any text at all! (data construction failure)
    if len(temp) != 0:
        return "".join(temp)
    else:
        return None
    

def get_raw_dataset(raw_table_path):
    """
    + looks for raw_dataset dir, if exist then gonna load it and return.
    + if not exist will create one and return:
        - there are 3 files (splits): trwiki-67.test.raw, trwiki-67.train.raw, trwiki-67.val.raw
        - go for every file
        - finds titles and doc-examples
        - concat everyting into one dict object
        - there are problematic examples (no text(null) or some lines of examples are not actually paragraph
        - but subtitles)
        - apply preprocess to handle these problems (delete subtitles and mask-filter-delete null examples)
        - measure length of examples and return as a new column
        - create dataset object and save it as .arrow format
        - and return
    """
    if not raw_table_path.startswith("data/tr_wiki67"):
        # if this is called from this script (prepare.py)
        prefix = "trwiki-67."
    else:
        # if this is called from outside
        prefix = "data/tr_wiki67/trwiki-67."

    # if raw dataset (.arrow) already exists
    if os.path.exists(raw_table_path):
        return load_from_disk(raw_table_path)
    
    # if dataset does not already exist, will be created
    print("raw dataset is not already exist, it will be created...")
    
    file_paths = [ f"{prefix}{post_fix}.raw" for post_fix in ["test", "train", "val"] ]
    print(file_paths)
    data_dict = {"examples":[], "titles":[]}


    # go for every file
    for file in file_paths:

        # open file
        with open(file, mode="+r", encoding="utf-8") as f:

            # read its content
            content = f.read()

            # with regular exp. find titles, examples(doc)
            examples = list(map(str.strip, re.split("== .* ==", content)[1:]))
            titles = list(re.findall("== .* ==", content))

            # package these with dict object
            temp_data_dict = {"examples":examples, "titles":titles}

        # concat dict objects from diffrent files into one overall dict object
        data_dict["examples"].extend(temp_data_dict["examples"])
        data_dict["titles"].extend(temp_data_dict["titles"])

    # convert data_dict object into dataframe object
    df = pd.DataFrame(data_dict)

    # we dont need these anymore, free memory
    del data_dict
    del temp_data_dict

    # lets delete lines of examples that are subtitles
    df["examples"] = df["examples"].map(__delete_subtitles)
    # lets also delete-mask df by null values (examples that does not contain text at all)
    mask = df["examples"].isnull()
    df = df[~mask]
    # lets measure length of examples (by char)
    df["length"] = df["examples"].map(len)
    # create hf dataset and save to disk
    df = df.reset_index(drop=True)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(raw_table_path)
    return dataset


def __create_pairs(raw_row):
    # split example by sentences
    list_of_sentences = sent_tokenize(raw_row["examples"], language="turkish")
    A = []
    B = []

    for pair_index in range(0, len(list_of_sentences)-1):
        A.append(list_of_sentences[pair_index])
        B.append(list_of_sentences[pair_index + 1])

    return {"A": A, "B": B}


def get_ab_table():

    if os.path.exists(ab_table_path):
        print("ab_table is already exists...")
        return load_from_disk(ab_table_path)

    # load raw dataset from disk
    raw_table = get_raw_dataset(raw_table_path)

    print("ab_table is not exists, will be created...")

    # # bu satır geçici! test'i hızlı yapabilmek için
    # raw_table = Dataset.from_dict(raw_table[:5_000])
    # every example needs at least 2 sentence
    raw_table_filtered = raw_table.filter(lambda row:  len(sent_tokenize(row["examples"], language="turkish")) > 1)


    # create table that contains A, B kols (problem: row of this table has list of sentences not just one sentence for each pair)
    a_b_list_table = raw_table_filtered.map(__create_pairs)


    # every element of A or B is diffrent list of strings, we are gonna join them (just one list of string object for each column)
    A, B = [], []
    for A_list, B_list in zip(a_b_list_table["A"], a_b_list_table["B"]):
        A += A_list
        B += B_list
    
    a_b_table = {"A":A, "B":B}
    a_b_table = Dataset.from_dict(a_b_table)

    a_b_table.save_to_disk(ab_table_path)

    return a_b_table




# i just want these if this file executed as a script but not import
if __name__ == "__main__":
    # GLOBAL VAR'S
    block_size = 255    # default val
    print(tokenizer_path)
    tokenizer = Tokenizer.from_file(tokenizer_path)   

    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    pad_id = tokenizer.token_to_id("[PAD]")
    mask_id = tokenizer.token_to_id("[MASK]")   

    a_b_table = get_ab_table()    


    # this chunk will be used for finding random word or sentence
    sentences_list = a_b_table[:10_000]
    sentences_list = sentences_list["A"] + sentences_list["B"]  # list + list -> list   



def get_random_sentence():
    return sentences_list[random.randint(0, len(sentences_list) - 1)]

def get_random_word_tokens(num_tokens):
    limit = 100
    trial = 0
    pattern = r'\d'

    while trial < limit:
        trial += 1
        words = get_random_sentence().split(" ")
        for word in words:
            # if word is number like 1881, 2005, ...
            # jumpt to next word
            if re.search(pattern, word):
                # print("number!! ", word)
                continue

            cand_word_token_ids = tokenizer.encode(word, add_special_tokens=False).ids
            if len(cand_word_token_ids) == num_tokens:
                # print(word)
                return cand_word_token_ids
            
    # trial limit exceeded return None
    return None


def __del_none_filter(row):
    return row["X"] != None

def __create_nsp_map(ab_row):
    
    isNext = random.random() > 0.5
    x = []  # input sequence
    y = []  # target sequence
    
    if isNext == False:
        rand_b = get_random_sentence()
        ab_row["B"] = rand_b

    encoding = tokenizer.encode(ab_row["A"], ab_row["B"])
    ab_len = len(encoding.tokens) 


    if ab_len > block_size:
        return {"X": None, "Y": None, "Attention_mask": None, "Token_type_id":None, "isNext": None}
    
    seq = tokenizer.decode(encoding.ids, skip_special_tokens=False).split(" ")

    for word in seq:

        if word == "[CLS]":
            x.append(cls_id)
            y.append(int(isNext))

        elif word == "[SEP]":
            x.append(sep_id)
            y.append(pad_id)
        
        else:
            tokens = tokenizer.encode(word, add_special_tokens=False).ids
            
            prob = random.random()
            if prob < 0.15 :


                prob_inner = random.random()
                if prob_inner < 0.8:
                    # mask condition
                    x += [mask_id for token in tokens]
                    y += tokens
                    # print("mask condition: ")
                    # print("x: ", tokenizer.decode([mask_id for token in tokens], skip_special_tokens=False))
                    # print("y: ", tokenizer.decode(tokens, skip_special_tokens=False))
                               
                elif prob_inner < 0.9:
                    # corrupt condition
                    random_word_tokens = get_random_word_tokens(num_tokens=len(tokens))

                    if random_word_tokens is None:
                        # could not found word that has the same token length as corrupted word
                        # print("could not found word")
                        # print("len of tokes: ", len(tokens))
                        return {"X": None, "Y": None, "Attention_mask": None, "Token_type_id":None, "isNext": None}
                    
                    x += random_word_tokens
                    y += tokens
                    # print("corrupt condition:")
                    # print("x: ", tokenizer.decode(random_word_tokens, skip_special_tokens=False))
                    # print("y: ", tokenizer.decode(tokens, skip_special_tokens=False))
                
                else:
                    # identity condition
                    x += tokens
                    y += tokens
                    # print("identity condition:")
                    # print("x: ", tokenizer.decode(tokens, skip_special_tokens=False))
                    # print("y: ", tokenizer.decode(tokens, skip_special_tokens=False))
                    
                
            else:
                x += tokens 
                y += [pad_id for token in tokens]
            

    assert len(x) == len(y), "[ERROR] there is some mistake, x and y should be the same length!"

    num_pad = block_size - len(x)
    x += [pad_id for each in range(num_pad)]
    y += [pad_id for each in range(num_pad)]
    
    first_sep_idx = x.index(sep_id)
    token_type_id = [ 0 if i <= first_sep_idx else 1 for i in range(len(x)) ]
    attention_mask = [ 0 if token_id == pad_id else 1 for token_id in x ]

    # print("\nX: ", tokenizer.decode(x, skip_special_tokens=False),
    #        "\nY: ", tokenizer.decode(y, skip_special_tokens=False),
    #        "\nAttention_mask: ", attention_mask, 
    #        "\nToken_type_id: ", token_type_id, 
    #        "\nisNext: ", isNext , "\n\n")
    
    # convert Tensor
    x_tensor = torch.tensor(x, dtype=torch.int16)
    y_tensor = torch.tensor(y, dtype=torch.int16)
    att_mask_tensor = torch.tensor(attention_mask, dtype=torch.int16)
    token_type_tensor = torch.tensor(token_type_id, dtype=torch.int16)
    isNext_tensor = torch.tensor(isNext, dtype=torch.int16)

    return {"X": x_tensor, "Y": y_tensor, "Attention_mask": att_mask_tensor, "Token_type_id":token_type_tensor, "isNext": isNext_tensor}

def get_nsp_mlm_table():

    if os.path.exists(nsp_mlm_table_path):
        print("nsp_mlm_table is already exists...")
        return load_from_disk(nsp_mlm_table_path)

    else:
        print("nsp_mlm_table is not exists, will be created...")
        nsp_mlm_table = a_b_table.map(lambda row: __create_nsp_map(row, )).filter(__del_none_filter).remove_columns(["A", "B"])
        nsp_mlm_table.save_to_disk(nsp_mlm_table_path)
        return nsp_mlm_table



def main():
    
    
    global block_size

    # command line arg (there is just block_size)
    if "--block_size=" in sys.argv[1]:
        # override global block_size
        block_size = int(sys.argv[1].split("=")[1])
        
    _ = get_nsp_mlm_table()




if __name__ == "__main__":
    main()