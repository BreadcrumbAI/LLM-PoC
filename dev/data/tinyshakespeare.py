"""
Downloads and tokenizes the TinyShakespeare dataset.
- The download is from Github.
- The tokenization is GPT-2 tokenizer with tiktoken

The output is written to a newly created tinyshakespeare/ folder.
The script prints:

Saved 32768 tokens to tinyshakespeare/tiny_shakespeare_val.bin
Saved 305260 tokens to tinyshakespeare/tiny_shakespeare_train.bin

And runs in a few seconds depending on your internet
connection and computer. The .bin files are raw byte
streams of int32 numbers indicating the token ids.
"""

import os
import tiktoken
import numpy as np
from data_common import download_file, write_datafile

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinyshakespeare")
gpt2_base = tiktoken.get_encoding("gpt2")
SPECIAL_TOKENS={'<|endoftext|>','<|br|>','<|er|>'}
DOC_ID = 1 


def encode(s):
    return gpt2_base.encode(s, allowed_special=SPECIAL_TOKENS)

def download():
    """Downloads the TinyShakespeare dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    # download the TinyShakespeare dataset, unless it's already downloaded
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

def save_better_text(text):
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_annotated.txt")
    print(f"Writing {data_filename} ...")
    with open(data_filename, 'wt') as bt:
        bt.write(text)
    

def makeTextBetter(text, docId=DOC_ID):
    yield '<|endoftext|>'
    counter = 1
    for part in text.split('\n\n'):
        docRef = f"TS-{docId}-{counter}"
        yield '<|br|>'
        yield docRef
        yield '<|er|>'
        yield ' '
        yield part
        yield '\n\n'
        yield '<|endoftext|>'
        counter += 1
  

def tokenize():
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")
    data_filename2 = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_new.txt")

    text1 = open(data_filename, 'r').read()
    text1 = ''.join(makeTextBetter(text1))

    text2 = open(data_filename2, 'r').read()
    text2 = ''.join(makeTextBetter(text2, 2))

    # Complete corpus:
    text = "\n\n".join([text1, text2])
    save_better_text(text)

    # encode the text
    tokens = encode(text)
    # let's take the first 32,768 tokens as the validation split (~10%)
    val_tokens = tokens[:32768]
    train_tokens = tokens[32768:]
    # save to file
    val_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_val.bin")
    train_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_train.bin")
    write_datafile(val_filename, val_tokens)
    write_datafile(train_filename, train_tokens)

if __name__ == "__main__":
    download()
    tokenize()
