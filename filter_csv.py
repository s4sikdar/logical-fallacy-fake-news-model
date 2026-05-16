import pandas as pd
from pathlib import Path


def politifact_filter_txt_file_exists(row):
    row_fname = row['article_filename']
    fname_components = row_fname.split('.')
    basename = fname_components[0]
    txt_path = ''
    if basename.endswith('true'):
        txt_path = f'./FakeNewsNet/dataset/politifact_real/{basename}.txt'
    if basename.endswith('false'):
        txt_path = f'./FakeNewsNet/dataset/politifact_fake/{basename}.txt'
    if txt_path == '':
        return False
    fname_path = Path(txt_path)
    if fname_path.exists():
        return True
    else:
        return False


def gossipcop_filter_txt_file_exists(row):
    row_fname = row['article_filename']
    fname_components = row_fname.split('.')
    basename = fname_components[0]
    txt_path = ''
    if basename.endswith('true'):
        txt_path = f'./FakeNewsNet/dataset/gossipcop_real/{basename}.txt'
    if basename.endswith('false'):
        txt_path = f'./FakeNewsNet/dataset/gossipcop_fake/{basename}.txt'
    if txt_path == '':
        return False
    fname_path = Path(txt_path)
    if fname_path.exists():
        return True
    else:
        return False


def assign_label(row):
    row_fname = row['article_filename']
    fname_components = row_fname.split('.')
    basename = fname_components[0]
    label = 'False'
    if basename.endswith('true'):
        label = 'True'
    if basename.endswith('false'):
        label = 'False'
    row['label'] = label
    return row


if __name__ == '__main__':
    gossipcop = pd.read_csv('gossipcop.csv')
    politifact = pd.read_csv('politifact.csv')
    gossipcop['label'] = ''
    politifact['label'] = ''
    gossipcop = gossipcop.loc[gossipcop.apply(gossipcop_filter_txt_file_exists, axis=1)]
    politifact = politifact.loc[politifact.apply(politifact_filter_txt_file_exists, axis=1)]
    gossipcop = gossipcop.apply(assign_label, axis=1)
    politifact = politifact.apply(assign_label, axis=1)
    gossipcop = gossipcop[['id', 'label', 'article_filename']]
    politifact = politifact[['id', 'label', 'article_filename']]
    gossipcop.to_csv('gossipcop_filtered.csv')
    politifact.to_csv('politifact_filtered.csv')
