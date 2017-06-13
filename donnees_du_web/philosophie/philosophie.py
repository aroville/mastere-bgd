#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Ne pas se soucier de ces imports
import setpath
from flask import Flask, render_template, session, request, redirect, flash
from getpage import getPage

app = Flask(__name__)
app.secret_key = "TODO: mettre une valeur secrète ici"

@app.route('/', methods=['GET'])
def index():
    session['title'] = None
    session['score'] = 0
    return render_template('index.html')


# Si vous définissez de nouvelles routes, faites-le ici
@app.route('/new-game', methods=['POST'])
def new_game():
    session['title'] = request.form['title']
    session['score'] = 0
    return redirect('/game')


@app.route('/game', methods=['GET'])
def game():
    next_title = session['title']
    if len(next_title) == 0:
        flash('you have to start somewhere !', 'danger')
        return redirect('/')

    title, links = getPage(next_title)
    if title is None:
        flash('no results !', 'danger')
        return redirect('/')

    if title.lower() == 'philosophie':
        flash('score: {}'.format(session['score']), 'success')
        return redirect('/')

    if not links:
        flash('no links available, back to the start !', 'danger')
        return redirect('/')

    return render_template('game.html', title=title, links=links)



@app.route('/move', methods=['POST'])
def move():
    if request.form['action'] == 'Restart':
        return redirect('/')

    next_title = request.form['next']
    previous_score = int(request.form['previous_score'])
    scores_match = previous_score == session['score']
    if not scores_match:
        titles_match = next_title == session['title']
        if not titles_match:
            flash('progression in another tab !', 'danger')
            return redirect('/game')

    if not next_title in getPage(session['title'])[1]:
        flash('invalid move !', 'danger')
        return redirect('/game')

    session['score'] += 1
    session['title'] = next_title

    return redirect('/game')


if __name__ == '__main__':
    app.run(threaded=True)
