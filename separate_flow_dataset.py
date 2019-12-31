import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import csv

### settings ####################################
output_dir = 'dataset/'
whole_file = Path(output_dir) / Path('flow_labeled.csv')
train_file = Path(output_dir) / Path('flow_labeled_train.csv')
val_file = Path(output_dir) / Path('flow_labeled_val.csv')
test_file = Path(output_dir) / Path('flow_labeled_test.csv')
# THE TARGET WRITER FOR STYLE TRANSFER
target_writer = 'ogawa'
target2_writer = 'akutagawa'
##############################################

# base {}_sample.csv
# ogawa shuffle 6 2 2
# akutagawa shuffle 6 2 2
# pathlib.Path({ogawa, akutagawa}).mkdir(exists_ok=True)
#
# {}_train.csv
# {}_val.csv
# {}_test.csv
# pandas
# header = ogawa akutagawa sentence
# concat
# shuffle


def load_target_csv(target: str, dir: Path):
    target_path: Path = dir / Path('{}_sample.csv'.format(target))
    if target_path.exists():
        df = pd.read_csv(target_path, header=None)
        df[target] = 1
        return df
    else:
        raise Exception('target is not found')


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
    tdf = load_target_csv(target_writer, Path(output_dir))
    t2df = load_target_csv(target2_writer, Path(output_dir))
    train_tds, val_tds, test_tds = split_t_v_t(tdf)
    train_t2ds, val_t2ds, test_t2ds = split_t_v_t(t2df)
    train_ds = merge_df([train_tds, train_t2ds])
    val_ds = merge_df([val_tds, val_t2ds])
    test_ds = merge_df([test_tds, test_t2ds])
    whole_ds = merge_df([tdf, t2df])
    whole_ds.to_csv(whole_file,
                    index=False,
                    encoding='utf-8',
                    quoting=csv.QUOTE_NONNUMERIC)
    train_ds.to_csv(train_file,
                    index=False,
                    encoding='utf-8',
                    quoting=csv.QUOTE_NONNUMERIC)
    val_ds.to_csv(val_file,
                  index=False,
                  encoding='utf-8',
                  quoting=csv.QUOTE_NONNUMERIC)
    test_ds.to_csv(test_file,
                   index=False,
                   encoding='utf-8',
                   quoting=csv.QUOTE_NONNUMERIC)
    print('[Info] file was saved at')
    print('[Info] whole: {} ({} sentences)'.format(whole_file, len(whole_ds)))
    print('[Info] train: {} ({} sentences)'.format(train_file, len(train_ds)))
    print('[Info] val: {} ({} sentences)'.format(val_file, len(val_ds)))
    print('[Info] test: {} ({} sentences)'.format(test_file, len(test_ds)))


if __name__ == '__main__':
    main()
