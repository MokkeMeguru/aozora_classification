import pandas as pd
from pathlib import Path

### settings ####################################
input_dir = 'aozora_data_integ/'
output_dir = 'dataset/'
# THE TARGET WRITER FOR STYLE TRANSFER
target_writer = 'ogawa'
target2_writer = 'akutagawa'
sample_size = 15000
##############################################


def load_data_without_target():
    dfs = []
    for path in Path(input_dir).glob('*.csv'):
        if path.name.startswith(target_writer) or path.name.startswith(
                target2_writer):
            continue
        else:
            dfs.append(pd.read_csv(path, index_col=0))
    df = pd.concat(dfs, ignore_index=True)
    return df


def sample_data(df, sample_size: int = 10000):
    sampled_df = df.sample(sample_size)
    return sampled_df, df.drop(sampled_df.index)


def sample_target_data(path: Path, sample_size: int = 10000):
    df = pd.read_csv(path, index_col=0)
    return sample_data(df, sample_size)


def main():
    t1_sampled_df, t1_df = sample_target_data(
        Path('aozora_data_integ/' + target_writer + '_integ.csv'), sample_size)
    print("[info] sampled_dataset {}: other-dataset {}".format(
        len(t1_sampled_df), len(t1_df)))

    t2_sampled_df, t2_df = sample_target_data(
        Path('aozora_data_integ/' + target2_writer + '_integ.csv'), sample_size)
    print("[info] sampled_dataset {}: other-dataset {}".format(
        len(t2_sampled_df), len(t2_df)))

    without_target_df = load_data_without_target()
    print("[info] dataset without target_df {}".format(len(without_target_df)))
    # t2_sampled_df, t2_df = sample_data(without_target_df)

    output_file = Path(output_dir + 'lm_dataset.csv')
    df = pd.concat([t1_df, t2_df, without_target_df], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df['line'].to_csv(output_file, index=False, encoding='utf-8', header=False)

    t1_output_file = Path(output_dir + target_writer + '_sample.csv')
    t1_sampled_df = t1_sampled_df.sample(frac=1).reset_index(drop=True)
    t1_sampled_df['line'].to_csv(t1_output_file,
                                 index=False,
                                 encoding='utf-8',
                                 header=False)

    t2_output_file = Path(output_dir + target2_writer + '_sample.csv')
    t2_sampled_df = t2_sampled_df.sample(frac=1).reset_index(drop=True)
    t2_sampled_df['line'].to_csv(t2_output_file,
                                 index=False,
                                 encoding='utf-8',
                                 header=False)


if __name__ == '__main__':
    main()

    # ogawa_sampled, ogawa_lm -> save
    # natsume_sampled, natsume_lm -> save

    # combined_data
    # else ogawa + _integ.csv
    # df.sample(frac=1).reset_index(drop=True)
