import pandas as pd
from pathlib import Path
from typing import List

lm_files = [Path("dataset/lm_dataset.csv")]
target_files = [Path("dataset/akutagawa_sample.csv"), Path("dataset/ogawa_sample.csv")]
save_file = Path ("dataset/sentence_pieces.csv")

def read_texts(path: Path):
    data = pd.read_table(path, header=None, encoding='utf-8')
    return data

def concat_texts(paths: List[Path]):
    data_list = [pd.read_table(path, header=None, encoding='utf-8') for path in paths]
    return pd.concat(data_list, )

def statistics_df ():
    df = concat_texts(lm_files + target_files)
    df ['length'] = df [0].map (lambda sentence: len (sentence))
    print ('[Info] whole sentence: {}'.format (len (df)))
    print ("[Stat] sentence length: mean {}".format (df ['length'].mean ()) )
    print ("[Stat] sentence legnth: std {}".format (df ['length'].std ()))
    return df
    
if __name__ == '__main__':
    df = statistics_df ()
    df [0].to_csv (save_file, index=False, encoding='utf-8', header=False)
    print ("[Info] for sentence pieces dataset was saved at {}".format (save_file))
