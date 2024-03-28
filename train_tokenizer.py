"""
python train_tokenizer.py
"""
import os
from data.tr_wiki67.prepare import create_tr_wiki_ds


def hello_from_train_tokenizer():
    print("hello from train tokenizer!")








def main():
    # hf dataset object
    dataset = None

    # initialize hf dataset by looking spesific dataset_path file
    dataset_path = "./data/tr-wiki67/tr-wiki67.arrow"
    if os.path.exists(dataset_path):
        print(f"[INFO] {dataset_path.split("/")[-1]} file is already exist...")
        dataset = ...
    else:
        print(f"[INFO] {dataset_path.split("/")[-1]} file is not exist, gonna create...")
        dataset = create_tr_wiki_ds()




if __name__ == "main":
    main()