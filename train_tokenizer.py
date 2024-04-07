"""
python train_tokenizer.py --vocab_size=25_000 --limit_alphabet=5 --save_path="wordpiece_tokenizer"
"""
import os
import sys
from data.tr_wiki67.prepare import get_raw_dataset
from dataclasses import dataclass, asdict
from ast import literal_eval
# from transformers import PreTrainedTokenizerFast

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

raw_dataset_path = "data/tr_wiki67/raw_dataset"


def get_example_batch(dataset):
    """
    generator func
    will be used when training tokenizer
    """
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["examples"]


@dataclass
class TokenizerConfig:
    vocab_size: int = 25_000
    limit_alphabet: int = 500
    min_frequency: int = 5
    save_path: str = "wordpiece_tokenizer"



def build_and_train_tokenizer(train_cfg:TokenizerConfig):

    dataset = get_raw_dataset(raw_dataset_path)

    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFKC(),
         normalizers.Lowercase()]
    )

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), 
                                               pre_tokenizers.Digits(individual_digits=True),
                                               pre_tokenizers.Punctuation()])
    
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

    trainer = trainers.WordPieceTrainer(vocab_size=train_cfg.vocab_size, special_tokens=special_tokens, 
                                        min_frequency=train_cfg.min_frequency,
                                        continuing_subword_prefix="##", 
                                        limit_alphabet=train_cfg.limit_alphabet)
    print("training...")

    tokenizer.train_from_iterator(get_example_batch(dataset), trainer=trainer)

    print("tokenizer training finished...")

    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]") 

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)]
        )
    
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    return tokenizer

def parse_args() -> TokenizerConfig:
    default_cfg = asdict(TokenizerConfig())
    
    for arg in sys.argv[1:]:
        assert arg.startswith("--")
        key, val = arg.split("=")
        key = key[2:]

        if key in default_cfg.keys():
            try:
                attempt = literal_eval(val)
            except(SyntaxError, ValueError):
                attempt = val
            
            assert type(attempt) == type(default_cfg[key]), f"type mismatch between {default_cfg[key]} type: {type(default_cfg[key])} and  {val} type: {type(val)}"
            default_cfg[key] = attempt
        
        else:
            raise ValueError(f"[ERROR] unknown arg key: {key}")
        

    # for test
    print(default_cfg)
    
    return TokenizerConfig(**default_cfg)



def main():
    train_cfg = parse_args() 

    if os.path.exists(train_cfg.save_path):
        print("tokenizer is already exists: ", train_cfg.save_path)

    else:
        

        tokenizer = build_and_train_tokenizer(train_cfg)
        print("tokenizer will be saved...")

        # wrapped_tokenizer = PreTrainedTokenizerFast(
        # tokenizer_object=tokenizer,
        # # tokenizer_file="tokenizer/tokenizer_55.json", # You can load from the tokenizer file, alternatively
        # unk_token="[UNK]",
        # pad_token="[PAD]",
        # cls_token="[CLS]",
        # sep_token="[SEP]",
        # mask_token="[MASK]",
        # )

        tokenizer.save(train_cfg.save_path)
    


if __name__ == "__main__":
    main()