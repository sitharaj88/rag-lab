# Syntax and Types

Think of this page as learning the alphabet before writing sentences. Python has a small set of core building blocks — once you know them, you can read almost any Python code you come across.

## What you'll learn

- Python's built-in types: `int`, `float`, `str`, `bool`, and `None`
- How to write f-strings and use operators
- Control flow with `if`, `for`, and `while`
- Python's truthiness rules
- How to read input and print output

## Variables and Assignment

A variable is like a labeled jar you put a value in. You write the label, then an equals sign, then the value. Python figures out what kind of value it is on its own — you do not need to declare a type upfront.

```python
name = "Alice"          # str  (text)
age = 30                # int  (whole number)
score = 9.5             # float (decimal number)
active = True           # bool (yes/no value)
result = None           # NoneType (the "nothing here" value)
```

You can change a variable's value at any time by assigning again. Variable names use `snake_case` (words joined by underscores) by convention.

## Core Built-in Types

Here is a quick summary of the five types you will use most often:

| Type    | Example            | Notes                          |
|---------|--------------------|--------------------------------|
| `int`   | `42`, `-7`         | Whole numbers, any size        |
| `float` | `3.14`, `-0.001`   | Numbers with a decimal point   |
| `str`   | `"hello"`, `'hi'`  | Text (the programming word is **string**) |
| `bool`  | `True`, `False`    | A yes/no, on/off value         |
| `None`  | `None`             | Represents "no value" or "nothing" |

You can ask Python what type a value is:

```python
print(type(42))             # <class 'int'>
print(isinstance(42, int))  # True
```

## Strings and f-strings

A string is just text wrapped in quotes. You can join strings together (the programming word for this is **concatenation**), but the modern way is to use an **f-string**, which lets you drop values directly into your text:

```python
greeting = "Hello"
subject = "world"

# Joining strings the old way
message = greeting + ", " + subject + "!"

# f-string — the modern, readable way
message = f"{greeting}, {subject}!"
print(message)  # Hello, world!

# You can put expressions inside the curly braces too
print(f"2 + 2 = {2 + 2}")          # 2 + 2 = 4
print(f"Upper: {subject.upper()}")  # Upper: WORLD
```

!!! tip "Stick with f-strings"
    f-strings (available since Python 3.6) are the modern standard. They are easier to read and faster to run than older approaches like `%` formatting or `.format()`.

## Operators

These symbols let you do maths and comparisons. The results are exactly what you would expect:

```python
# Arithmetic
print(10 + 3)   # 13
print(10 - 3)   # 7
print(10 * 3)   # 30
print(10 / 3)   # 3.3333...  (always gives a decimal)
print(10 // 3)  # 3          (whole-number division, drops the remainder)
print(10 % 3)   # 1          (just the remainder)
print(2 ** 8)   # 256        (2 to the power of 8)

# Comparison (these give back True or False)
print(5 > 3)    # True
print(5 == 5)   # True
print(5 != 4)   # True

# Logical (combining True/False values)
print(True and False)  # False
print(True or False)   # True
print(not True)        # False
```

## Control Flow

Control flow is how you tell Python to make decisions or repeat steps.

### `if` / `elif` / `else`

This block checks a condition and runs different code depending on whether it is true. Think of it as an "if this, do that, otherwise do something else" instruction:

```python
temperature = 72

if temperature > 90:
    print("Hot day")
elif temperature > 60:
    print("Nice day")
else:
    print("Cold day")
# Nice day
```

### `for` loops

A `for` loop repeats a block of code once for each item in a collection. Think of it as "for each item in my list, do this":

```python
chunks = ["chunk_1", "chunk_2", "chunk_3"]

for chunk in chunks:
    print(f"Processing: {chunk}")

# You can also loop over a range of numbers
for i in range(3):
    print(i)   # 0, 1, 2
```

### `while` loops

A `while` loop keeps repeating as long as a condition is true. It stops as soon as the condition becomes false:

```python
count = 0
while count < 3:
    print(f"Count: {count}")
    count += 1
```

!!! tip "Indentation is not decoration — it is the code"
    Python uses indentation (4 spaces) to show what is inside a block. There are no curly braces like in some other languages. If your indentation is inconsistent, Python will stop and tell you about it with a `SyntaxError`. Most editors handle this for you automatically.

## Truthiness

Python has a neat shortcut: many values can act as `True` or `False` even when they are not booleans (the programming word for this is **truthiness**). These values are treated as `False`:

```python
bool(0)       # False  (the number zero)
bool(0.0)     # False  (zero as a decimal)
bool("")      # False  (an empty string)
bool([])      # False  (an empty list)
bool(None)    # False  (the "nothing" value)

# Everything else is treated as True
bool(1)       # True
bool("hi")    # True
bool([0])     # True  (a list with something in it)
```

This lets you write clean, readable checks like this one:

```python
text = ""
if not text:
    print("No text provided!")
```

## Input and Print

`print()` sends text to your terminal. `input()` asks the user to type something:

```python
# Print with custom separator and ending
print("Hello", "World", sep=", ", end="!\n")  # Hello, World!

# Ask the user to type something (always comes back as a string)
name = input("Enter your name: ")
print(f"Hi, {name}!")
```

!!! tip "You will rarely use `input()` in RAG projects"
    In scripts and data pipelines, you normally get your configuration from files or environment variables rather than typing it in. Do not worry if you never use `input()` — it is good to know it exists.

!!! example "Try it yourself"
    Open the Python REPL (`python` in your terminal) and type these lines one at a time. Watch what comes back after each one. Try changing the numbers and strings to see what changes in the output.

    ```python
    x = 10
    y = 3
    print(f"{x} divided by {y} is {x / y:.2f}")
    ```

    The `:.2f` inside the curly braces is a format hint that rounds the decimal to two places. You will see `10 divided by 3 is 3.33`.

## Next steps

- [Data Structures](data-structures.md) — lists, dicts, sets, and comprehensions
- [Functions and Modules](functions-and-modules.md) — organising code into reusable units
