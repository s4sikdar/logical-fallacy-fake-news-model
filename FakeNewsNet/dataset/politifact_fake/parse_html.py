from bs4 import BeautifulSoup
from bs4.element import Comment
from pathlib import Path
import logging


logfile = 'politifact_fake_test.log'


def visible_tag(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def extract_text(html_fname):
    article_text = ''
    output_text_fname = html_fname.replace('.html', '.txt')
    with open(html_fname, encoding='latin-1') as html_file:
    #with open('politifact1013.html') as html_file:
        soup = BeautifulSoup(html_file, features="html.parser")
        texts = []
        soup_body = soup.body
        if soup_body:
            texts = soup_body.find_all(string=True)
        else:
            texts = soup.find_all(string=True)
        visible_texts = filter(visible_tag, texts)
        article_text = u" ".join(text.strip().replace('\t', ' ') for text in visible_texts)

    if article_text != '':
        with open(output_text_fname, 'w') as plaintext_output:
            #for text_line in soup.stripped_strings:
            #    plaintext_line = repr(text_line)
            #    print(plaintext_line, file=plaintext_output)
            print(article_text, file = plaintext_output)
        logging.info(f'parsed {html_fname} into {output_text_fname}')
    else:
        logging.error(f'failed to parse {html_fname} into {output_text_fname} since article text parsed from html file was an empty string')

if __name__ == '__main__':
    logging.basicConfig(
            filename=logfile,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt= '%Y-%m-%d %H:%M:%S',
            level = logging.INFO
    )
    current_dir = Path('.')
    html_files = list(current_dir.glob('*.html'))
    for html_filename in html_files:
        extract_text(str(html_filename.name))
