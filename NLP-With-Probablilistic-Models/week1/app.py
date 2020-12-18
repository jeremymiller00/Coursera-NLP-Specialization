#!/opt/anaconda3/bin/python

import click
from auto_correct import AutoCorrector


@click.command()
@click.argument('text', default='Enter some text')
def correct(text):
    ac = AutoCorrector()
    ac.create_vocab()
    ac.calculate_probabilities()
    words = text.split()
    corr_words = []
    for word in words:
        if ac.is_misspelled(word):
            ac.create_candidates(word)
            ac.filter_candidates()
            corr_words.append(ac.get_top_candidate())
        else:
            corr_words.append(word)
    corr = " ".join(corr_words)
    click.echo(f"Corrected: {corr}")


if __name__ == '__main__':
    correct()
