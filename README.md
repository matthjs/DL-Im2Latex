<br />
<p align="center">
  <h1 align="center">Image-to-LaTeX Converter for Mathematical Formulas and Text - im2latex</h1>

  <p align="center">
  </p>
</p>

## About The Project

Deep Learning Project on Transformer Optical Character Recognition (TrOCR). Specifically, generation of LaTeX equation code from a LaTeX image.

## Getting started

### Prerequisites
- [Docker v4.25](https://www.docker.com/get-started) or higher (if running docker container).
- [Poetry](https://python-poetry.org/).
## Running

Using docker: Run the docker-compose files to run all relevant services (`docker compose up` or `docker compose up --build`).

You can also set up a virtual environment using Poetry. Poetry can  be installed using `pip`:
```
pip install poetry
```
Then initiate the virtual environment with the required dependencies (see `poetry.lock`, `pyproject.toml`):
```
poetry config virtualenvs.in-project true    # ensures virtual environment is in project
poetry install
```
The virtual environment can be accessed from the shell using:
```
poetry shell
```
IDEs like Pycharm will be able to detect the interpreter of this virtual environment.

## Usage

See `main.py` for training or finetuning, `evaluate_models.py` for evaluation and `streamlit_app` for an inference GUI.

# License
This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](./LICENSE) file for details.
