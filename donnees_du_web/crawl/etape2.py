from os import makedirs
from os.path import join, exists
import requests as re
from tasktimer import call_repeatedly

FPO_URL = 'http://www.freepatentsonline.com/'
PATENTS_DIR = 'patents/'


def get_from_fpo(patent_ids):
    if not patent_ids:
        return True

    patent_id = patent_ids.pop(0)
    if not exists(PATENTS_DIR):
        makedirs(PATENTS_DIR)

    if '/' in patent_id:
        sub_dir_name = patent_id.split('/')[0]
        if not exists(join(PATENTS_DIR, sub_dir_name)):
            makedirs(join(PATENTS_DIR, sub_dir_name))

    url = FPO_URL + patent_id
    with open(join(PATENTS_DIR, patent_id), 'w') as f:
        f.write(re.get(url).text)

    return False


if __name__ == '__main__':
    call_repeatedly(5, get_from_fpo, [
        'y2015/0032669.html',
        'y2003/0132591.html'
    ])
