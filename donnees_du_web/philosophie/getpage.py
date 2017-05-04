#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Ne pas se soucier de ces imports
import setpath
from bs4 import BeautifulSoup
from json import loads
from urllib.request import urlopen
from urllib.parse import urlencode, unquote


# Si vous écrivez des fonctions en plus, faites-le ici
cache = {}

def getJSON(page):
    params = urlencode({
      'format': 'json',
      'action': 'parse',
      'prop': 'text',
      'redirects': 'true',
      'page': page})
    API = "https://fr.wikipedia.org/w/api.php"
    response = urlopen(API + "?" + params)
    return response.read().decode('utf-8')


def getRawPage(page):
    parsed = loads(getJSON(page))['parse']
    try:
        title = parsed['title']
        content = parsed['text']['*']
        return title, content
    except KeyError:
        return None, None


def internal_link(href):
    if not href:
        return False
    if '/wiki/' not in href:
        return False
    if 'redlink=1' in href:
        return False
    if 'API_' in href:
        return False
    if ':' in href:
        return False
    return True


def getPage(page):
    try:
        title, content = getRawPage(page)
    except KeyError as e:
        return None, []

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
                    hrefs.append(unquote(href))

    cache[title] = hrefs
    return title, hrefs


if __name__ == '__main__':
    # Ce code est exécuté lorsque l'on exécute le fichier
    # print("Ça fonctionne !")

    # Voici des idées pour tester vos fonctions :
    # pprint(getJSON("Utilisateur:A3nm/INF344"))
    pprint(getPage("Utilisateur:A3nm/INF344"))
    # pprint(getPage("philosophique"))
    # print(getRawPage("Histoire"))

