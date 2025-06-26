import marimo

__generated_with = "0.14.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Welcome to the Enverge""")
    return


@app.cell
def _(mo):
    mo.md(r"""## This is a blank Marimo notebook""")
    return


@app.cell
def _():
    from tqdm import tqdm

    from ollama import pull

    current_digest, bars = '', {}
    for progress in pull('llama3.2', stream=True):
      digest = progress.get('digest', '')
      if digest != current_digest and current_digest in bars:
        bars[current_digest].close()

      if not digest:
        print(progress.get('status'))
        continue

      if digest not in bars and (total := progress.get('total')):
        bars[digest] = tqdm(total=total, desc=f'pulling {digest[7:19]}', unit='B', unit_scale=True)

      if completed := progress.get('completed'):
        bars[digest].update(completed - bars[digest].n)

      current_digest = digest
    return


@app.cell
def _():
    from ollama import chat

    messages = [
      {
        'role': 'user',
        'content': 'Why is the sky blue on Tuesdays?',
      },
    ]

    response = chat('llama3.2', messages=messages)
    print(response['message']['content'])
    return


if __name__ == "__main__":
    app.run()
