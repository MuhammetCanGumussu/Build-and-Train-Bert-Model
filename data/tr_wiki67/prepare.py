"""
python prepare.py train_tokenizer
"""
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer

from datasets import DatasetDict, Dataset, load_from_disk
import pandas as pd
import re
import os


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
    

def create_and_save_tr_wiki_ds():
    """
    + there are 3 files (splits): trwiki-67.test.raw, trwiki-67.train.raw, trwiki-67.val.raw
    + this function goes for every file
    + finds titles and doc-examples
    + concat everyting into one dict object
    + there are problematic examples (no text(null) or some lines of examples are not actually paragraph
      but subtitles)
    + apply preprocess to handle these problems (delete subtitles and mask-filter-delete null examples)
    + measure length of examples and return as a new column
    + create dataset object and save it as .arrow format
    """
    prefix = "data/tr_wiki67/trwiki-67."
    file_paths = [ f"{prefix}{post_fix}.raw" for post_fix in ["test", "train", "val"] ]

    data_dict = {"examples":[], "titles":[]}

    

    # go for every file
    for file in file_paths:

        # open file
        with open(file) as f:

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
        dataset.save_to_disk("data/tr_wiki67/tr_wiki67_dataset")


def main():
    pass












if __name__ == "main":
    main()