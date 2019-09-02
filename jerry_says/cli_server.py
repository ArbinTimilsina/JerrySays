import os

from flask import Flask, request, render_template

from jerry_says.server import greedy_server

app = Flask(__name__)

PATH_TO_MODEL = os.path.join('trained_model', 'transformer-for-jerry.pt')


@app.route('/')
def get_seed():
    return render_template('for_seed.html')


@app.route('/completed', methods=['POST'])
def make_completions():
    if request.method == 'POST':
        seed_text = request.form['seed']
        completion = greedy_server(PATH_TO_MODEL, seed_text)

        return render_template(
            'render_this.html', seed=seed_text, jerry_says=completion
            )


def _main(host="localhost", port=5050):
    app.run(host=host, port=port, debug=False)


def main():
    """
    The setuptools entry point.
    """
    _main()
