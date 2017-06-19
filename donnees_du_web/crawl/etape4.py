import requests
from bs4 import BeautifulSoup
from tasktimer import call_repeatedly

FPO_URL = 'http://www.freepatentsonline.com/'
N = 50
links = []
visited = set()


def get_from_fpo(urls):
    if not urls:
        return True

    url = urls.pop(0)
    soup = BeautifulSoup(requests.get(FPO_URL + url).text, 'html.parser')

    for l in soup.find_all('a', href=True):
        if len(links) >= N:
            return True

        href = l.get('href')
        if not (href in visited or href in urls):
            print(href)
            links.append(href)
            urls.append(href)

    visited.add(url)
    return False


if __name__ == '__main__':
    call_repeatedly(5, get_from_fpo, [
        '/y2015/0032669.html',
        '/y2003/0132591.html'
    ])

