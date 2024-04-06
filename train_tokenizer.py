"""
python train_tokenizer.py
"""
import os
import sys
from data.tr_wiki67.prepare import get_raw_dataset
from dataclasses import dataclass, asdict
from ast import literal_eval

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)



dataset = get_raw_dataset()



def get_example_batch():
    """
    generator func
    will be used when training tokenizer
    """
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["examples"]


@dataclass
class TokenizerConfig:
    vocab_size: int = 25_000
    limit_alpabet: int = 500
    min_frequency: int = 5
    path_to_save: str = None



def build_and_train_tokenizer(train_cfg:TokenizerConfig):

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
                                        limit_alphabet=train_cfg.limit_alpabet)

    tokenizer.train_from_iterator(get_example_batch(), trainer=trainer)

    print("tokenizer training finished...")

    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]") 

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)]
        )
    
    tokenizer.decoder = decoders.WordPiece(prefix="##")


def parse_args() -> TokenizerConfig:
    default_cfg = asdict(TokenizerConfig())
    
    for arg in sys.argv[1:]:
        assert arg.startswith("--")
        key, val = arg.split("=")
        key = key[2:]

        if key in default_cfg.keys():
            val = literal_eval(val)
            assert type(val) == type(default_cfg[key])
            default_cfg[key] = val
        
        else:
            raise ValueError(f"[ERROR] unknown arg key: {key}")
        
    if default_cfg["path_to_save"] is None:
        # default path
        default_cfg["path_to_save"] = "./tokenizer_wordpiece"   
    
    return TokenizerConfig(**default_cfg)




        



def main():
    train_cfg = parse_args() 

    if os.path.exists(train_cfg.path_to_save):
        print("tokenizer is already exists...")

    else:
        tokenizer = build_and_train_tokenizer(train_cfg)
        print("tokenizer will be saved...")
        tokenizer.save_pretrained(train_cfg.path_to_save)
    



if __name__ == "main":
    main()