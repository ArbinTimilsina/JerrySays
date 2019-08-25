import json
import os

from flask import Flask, request

from jerry_says.server import greedy_server

app = Flask(__name__)

PATH_TO_MODEL = os.path.join('trained_model', 'transformer-for-jerry.pt')


@app.route("/jerry-says")
def make_completions():
    try:
        seed_text = request.args.get("seed")
        suggestions = greedy_server(PATH_TO_MODEL, seed_text)
        return json.dumps(
            {"Seed": seed_text, "Suggested completions": suggestions},
            indent=2
        )
    except Exception as e:
        print(e)


def _main(host="localhost", port=5050):
    app.run(host=host, port=port, debug=False)


def main():
    """
    The setuptools entry point.
    """
    _main()
