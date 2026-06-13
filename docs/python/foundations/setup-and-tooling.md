# Setup and Tooling

Think of this page as setting up your workshop before you start building. You would not cook a meal in a kitchen with no tools — and you would not write Python without a clean, organised environment. Let's get yours ready!

## What you'll learn

- How to install Python 3.12+ and check that it worked
- What a virtual environment is and why you always want one
- How to create and activate environments with both `python -m venv` and `uv`
- How to install packages and keep your code tidy with `ruff`
- How to run scripts and use the interactive REPL

## Installing Python 3.12+

Download the latest Python 3.12+ installer from [python.org](https://www.python.org/downloads/). On Windows, tick the box that says **"Add Python to PATH"** — this one step saves a lot of headaches later.

After installing, open a terminal and run this command to check everything worked:

```bash
python --version
# Python 3.12.x
```

You should see a version number printed back at you. That means Python is installed and ready.

!!! tip "On macOS and Linux, use `python3`"
    On macOS and Linux, typing `python` may open an old version called Python 2. Use `python3` and `pip3` to be safe. Or use `uv` (covered below) — it always picks the right version automatically.

## What is a Virtual Environment?

Imagine you have two cookbooks, and each one calls for a different brand of flour. If you only have one bag of flour in your kitchen, one recipe will always be wrong. A virtual environment (the programming word for this concept) is like giving each project its own private shelf of ingredients. Each project gets exactly the package versions it needs, and they never clash with each other.

!!! tip "Always use a virtual environment"
    Get into the habit of creating a virtual environment for every new project. It takes ten seconds and saves hours of debugging later.

## Option A — Classic `python -m venv`

This command creates a new virtual environment in a folder called `.venv` inside your project:

**Create and activate (PowerShell on Windows):**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Create and activate (bash/zsh on macOS/Linux):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Once activated, you will see `(.venv)` appear at the start of your terminal prompt. That tells you the environment is active and ready.

**When you are done working, deactivate it:**

```bash
deactivate
```

## Option B — `uv` (Recommended for Speed)

`uv` is a newer, much faster tool that handles virtual environments and package installation. Think of it as a turbo-charged version of the classic tools.

**Install `uv` first:**

```bash
pip install uv
# or on macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Then create and activate a virtual environment:**

```bash
uv venv .venv
# Activate exactly the same way as above:
.venv\Scripts\Activate.ps1   # PowerShell on Windows
source .venv/bin/activate     # bash/zsh on macOS/Linux
```

!!! tip "Why bother with `uv`?"
    `uv` installs packages 10–100x faster than the classic `pip`. It can also install specific Python versions for you with `uv python install 3.12`. For new projects, it is a great choice.

## Installing Packages

Packages are ready-made tools other people have built that you can add to your project. You install them from the internet with one command.

**With pip (classic):**

```bash
pip install requests
pip install -r requirements.txt
```

**With uv:**

```bash
uv pip install requests
uv pip install -r requirements.txt
```

**Save your list of dependencies so others can recreate your environment:**

```bash
pip freeze > requirements.txt      # classic
uv pip freeze > requirements.txt   # uv
```

## Keeping Your Code Tidy with `ruff`

`ruff` is a tool that reads your code and points out style problems or potential mistakes. Think of it as a helpful proofreader for your Python files. It replaces several older tools (`flake8`, `isort`, `black`) in one fast package.

Install it:

```bash
uv pip install ruff
# or
pip install ruff
```

**Format your code so it looks consistent:**

```bash
ruff format .
```

**Check for problems:**

```bash
ruff check .
```

**Let `ruff` fix the problems it can fix automatically:**

```bash
ruff check --fix .
```

!!! example "Try it yourself"
    Create a file called `hello.py`, write a few lines of Python, then run `ruff format hello.py`. Open the file and see if `ruff` changed any spacing or quotes. Most editors also have a `ruff` extension that tidies your code every time you save — well worth installing.

## Running Scripts

This command runs a Python file called `my_script.py`:

```bash
python my_script.py           # classic
uv run python my_script.py    # uv (works even without activating the environment first)
```

`uv run` is handy because it automatically finds and uses your project's virtual environment for you.

## The Python REPL

The REPL (which stands for Read–Eval–Print Loop — but you do not need to memorise that) is an interactive playground where you can type Python and see results instantly. It is great for experimenting.

Start it by typing:

```bash
python
```

Then try typing these lines one at a time:

```python
>>> 2 + 2
4
>>> name = "RAG"
>>> print(f"Hello, {name}!")
Hello, RAG!
>>> exit()
```

Each line you type gets evaluated and the result appears right below it. Type `exit()` when you want to leave.

!!! tip "A better REPL: ipython"
    Run `uv pip install ipython` and then start `ipython` instead of `python`. You get coloured output, tab completion, and the ability to run shell commands. It makes experimenting much more enjoyable.

## Next steps

- [Syntax and Types](syntax-and-types.md) — variables, strings, control flow
- [Environment Setup](../../getting-started/environment-setup.md) — project-level setup for the RAG lab
