import requests as re
from bs4 import BeautifulSoup
from tasktimer import call_repeatedly

FPO_URL = 'http://www.freepatentsonline.com/'
N = 50
links = []


def get_from_fpo(urls):
    if not urls:
        return True

    url = FPO_URL + urls.pop(0)
    soup = BeautifulSoup(re.get(url).text, 'html.parser')

    for l in soup.find_all('a'):
        if len(links) >= N:
            return True

        href = l.get('href')
        if href:
            print(href)
            links.append(href)
            urls.append(href)

    return False


if __name__ == '__main__':
    call_repeatedly(5, get_from_fpo, [
        '/y2015/0032669.html',
        '/y2003/0132591.html'
    ])

