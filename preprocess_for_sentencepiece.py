import sentencepiece as spm
import pandas as pd
from pathlib import Path
import numpy as np

##############################
default_symbols = ['<mask>']
vocab_f_path = Path('dataset/sentence_pieces.csv')
save_file_prefix = 'dataset/m'
##############################


def dump_dataframe(df, dst_path):
    """
    dump File

    Parameters
    ------
    df: pandas.DataFrame
         Returns from  concat_files
    dst_path: pathlib.Path
        output file path

    Returns
    ------
    """
    df.to_csv(dst_path,
              header=False,
              index=False,
              encoding='utf-8',
              sep='\t',
              mode='w')


def train_SentencePiece(fname,
                        prefix=save_file_prefix,
                        vocab_size=2000,
                        character_coverage=0.995,
                        model_type='unigram',
                        user_defined_symbols=default_symbols):
    """
    Training SentencePiece
    ref. https://github.com/google/sentencepiece

    Parameters
    ------
    fname: pathlib.Path
        csv's path

    prefix: str
        save file's prefix ex. `m`

    vocab_size: int
        vocaburary size

    character_coverage: float [0, 1]
        character_coverage
        default 1.

    model_type: str
        ['unigram', 'char', 'bpe', 'word']

    Returns
    ------
    """
    parameter = '--input={} --model_prefix={} --pad_id=3 --vocab_size={} --character_coverage={} --model_type={} --user_defined_symbols={}'.format(
        str(fname), prefix, vocab_size, character_coverage, model_type,
        ','.join(user_defined_symbols))
    spm.SentencePieceTrainer.Train(parameter)


def test_SentencePieces(example_sentence='世界よこんにちは。', prefix=save_file_prefix):
    sp = spm.SentencePieceProcessor()
    sp.Load(prefix + '.model')
    encodes = sp.EncodeAsPieces(example_sentence)
    print(encodes)

def gen_tokenizer(prefix):
    sp = spm.SentencePieceProcessor()
    sp.Load(prefix + '.model')
    return sp

def get_statistics(vocab_path=vocab_f_path, prefix=save_file_prefix):
    sp = spm.SentencePieceProcessor()
    sp.Load(prefix + '.model')
    df = pd.read_table(vocab_path, header=None, encoding='utf-8')
    df['length'] = df[0].map(lambda sentence: len(sp.EncodeAsPieces(sentence)))
    print('[Info] tokenized sentence length: max {}'.format(df['length'].max()))
    print('[Info] tokenized sentence length: min {}'.format(df['length'].min()))
    print('[Info] tokenized sentence length: mean {}'.format(df['length'].mean()))
    print('[Info] tokenized sentence length: std {}'.format(df['length'].std()))
    from pprint import pprint
    print('[debug] tokenized minimum sentences')
    pprint(df[df['length'] == df['length'].min()])
    print('[debug] tokenized minimum sentences')
    pprint(df[df['length'] == df['length'].max()])

def main():
    train_SentencePiece(vocab_f_path, vocab_size=8000)
    test_SentencePieces()
    get_statistics()

if __name__ == '__main__':
    main()
