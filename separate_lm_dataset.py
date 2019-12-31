import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import csv

### settings ####################################
output_dir = 'dataset/'
flow_train_file = Path(output_dir) / Path('flow_labeled_train.csv')
lm_file = Path(output_dir) / Path('lm_dataset.csv')
lm_train_file = Path(output_dir) / Path('lm_dataset_train.csv')
lm_val_file = Path(output_dir) / Path('lm_dataset_val.csv')
lm_test_file = Path(output_dir) / Path('lm_dataset_test.csv')
# THE TARGET WRITER FOR STYLE TRANSFER
target_writer = 'ogawa'
target2_writer = 'akutagawa'
##############################################

def split_t_v_t(ds, val=0.2, test=0.2):
    """split dataframe as
    6:2:2
    used by sklearn
    """
    ds = ds.sample(frac=1).reset_index(drop=True)
    train_ds, test_ds = train_test_split(ds, test_size=test)
    train_ds, val_ds = train_test_split(train_ds, test_size=(val / (1 - test)))
    return train_ds, val_ds, test_ds


def merge_df(dfs):
    df = pd.concat(dfs, sort=False)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.fillna(0)
    return df


def main():
    flow_train_df = pd.read_csv(flow_train_file, encoding='utf-8')['0']
    lm_df = pd.read_table(lm_file, header=None, encoding='utf-8')
    print('[Info] dataset was joined {}+{}={}'
          .format(len(flow_train_df), len(lm_df), len(flow_train_df) + len(lm_df)))
    lm_df = pd.concat([lm_df, flow_train_df])
    train_lm_df, val_lm_df, test_lm_df = split_t_v_t(lm_df, 0.1, 0.1)
    train_lm_df.to_csv(lm_train_file,
                       index=False,
                    encoding='utf-8',
                    quoting=csv.QUOTE_NONNUMERIC)
    val_lm_df.to_csv(lm_val_file,
                       index=False,
                     encoding='utf-8',
                     quoting=csv.QUOTE_NONNUMERIC)
    test_lm_df.to_csv(lm_test_file,
                       index=False,
                      encoding='utf-8',
                      quoting=csv.QUOTE_NONNUMERIC)
    print('[Info] file was saved at')
    print('[Info] train: {} ({} sentences)'.format(lm_train_file, len(train_lm_df)))
    print('[Info] val: {} ({} sentences)'.format(lm_val_file, len(val_lm_df)))
    print('[Info] test: {} ({} sentences)'.format(lm_test_file, len(test_lm_df)))

if __name__ == '__main__':
    main()
