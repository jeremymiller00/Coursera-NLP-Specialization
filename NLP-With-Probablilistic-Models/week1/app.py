#!/opt/anaconda3/bin/python

import click
from auto_correct import AutoCorrector


@click.command()
@click.option('--max_dist', default=2, help='Max string edit distance.')
@click.argument('text', default='Enter some text')
def correct(text, max_dist):
    ac = AutoCorrector(max_dist=max_dist)
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
