#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Ne pas se soucier de ces imports
from bs4 import BeautifulSoup
from json import loads
from urllib.parse import unquote, urlencode
import requests

# Si vous écrivez des fonctions en plus, faites-le ici
API = "https://fr.wikipedia.org/w/api.php"
params = {
  'format'      : 'json',
  'action'      : 'parse',
  'prop'        : 'text',
  'redirects'   : 'true'
}

cache = {}
redirects = {}

def getJSON(page):
    global params
    params['page'] = page
    return requests.get(API + '?' + urlencode(params)).text


def getRawPage(page):
    try:
        parsed = loads(getJSON(page))['parse']
        title = parsed['title']
        content = parsed['text']['*']
        return title, content
    except KeyError:
        return None, []


def internal_link(href):
    if not href or '/wiki/' not in href:
        return False
    return not('redlink=1' in href or 'API_' in href or ':' in href)


def getPage(page):
    if page in redirects.keys():
        page = redirects[page]

    if page in cache.keys():
        return page, cache[page]

    title, content = getRawPage(page)
    if title is None:
        return None, []

    redirects[page] = title
    if title in cache.keys():
        return title, cache[title]

    soup = BeautifulSoup(content, 'html.parser')

    found10 = False
    hrefs = []
    for p in soup.find_all('p', recursive=False):
        if found10:
            break

        for link in p.find_all('a'):
            if len(hrefs) >= 10:
                found10 = True
                break

            href = link.get('href')
            if internal_link(href):
                href = href.split('/wiki/')[-1]
                href = href.split('#')[0]
                href = unquote(href)
                if href not in hrefs:
                    hrefs.append(href)

    cache[title] = hrefs
    return title, hrefs


if __name__ == '__main__':
    # Ce code est exécuté lorsque l'on exécute le fichier
    # print("Ça fonctionne !")

    # Voici des idées pour tester vos fonctions :
    # pprint(getJSON("Utilisateur:A3nm/INF344"))
    print(getPage("Utilisateur:A3nm/INF344"))
    # pprint(getPage("philosophique"))
    # print(getRawPage("Histoire"))

