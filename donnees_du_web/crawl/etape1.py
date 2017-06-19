from os import makedirs
from os.path import join, exists
import requests as re

FPO_URL = 'http://www.freepatentsonline.com/'
PATENTS_DIR = 'patents/'


def get_from_fpo(patent_id):
    if not exists(PATENTS_DIR):
        makedirs(PATENTS_DIR)

    url = FPO_URL + patent_id

    if '/' in patent_id:
        sub_dir_name = patent_id.split('/')[0]
        if not exists(join(PATENTS_DIR, sub_dir_name)):
            makedirs(join(PATENTS_DIR, sub_dir_name))

    with open(join(PATENTS_DIR, patent_id), 'w') as f:
        f.write(re.get(url).text)

if __name__ == '__main__':
    get_from_fpo('y2015/0032669.html')
