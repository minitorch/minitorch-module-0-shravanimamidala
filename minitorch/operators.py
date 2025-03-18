import math
from typing import Callable, Iterable
from typing import List

# ## Task 0.1

# TODO: Implement for Task 0.1.

# Mathematical functions:

def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y

def prod(lst: List[float]) -> float:
    result = 1.0
    for num in lst:
        result *= num
    return result

def id(x: float) -> float:
    """Identity function."""
    return x

def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y

def neg(x: float) -> float:
    """Negate a number."""
    return -x

def lt(x: float, y: float) -> bool:
    """Check if x is less than y."""
    return x < y

def eq(x: float, y: float) -> bool:
    """Check if x is equal to y."""
    return x == y

def max(x: float, y: float) -> float:
    """Return the maximum of two numbers."""
    return x if x > y else y

def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close within a small tolerance (1e-2)."""
    return abs(x - y) < 1e-2

def sigmoid(x: float) -> float:
    """Compute the sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))

def relu(x: float) -> float:
    """ReLU activation function."""
    return max(0, x)

def log(x: float) -> float:
    """Compute the natural logarithm of x."""
    return math.log(x)

def exp(x: float) -> float:
    """Compute the exponential function e^x."""
    return math.exp(x)

def log_back(x: float, d: float) -> float:
    """Compute the gradient of the log function."""
    return d / x

def inv(x: float) -> float:
    """Compute the inverse (1/x)."""
    return 1.0 / x

def inv_back(x: float, d: float) -> float:
    """Compute the gradient of the inverse function."""
    return -d / (x ** 2)

def relu_back(x: float, d: float) -> float:
    """Compute the gradient of ReLU."""
    return d if x > 0 else 0


# ## Task 0.3

# TODO: Implement for Task 0.3.

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce

def map(fn: Callable[[float], float], lst: Iterable[float]) -> list:
    """Apply a function to each element in a list."""
    return [fn(x) for x in lst]

def zipWith(fn: Callable[[float, float], float], lst1: Iterable[float], lst2: Iterable[float]) -> list:
    """Apply a function to pairs of elements from two lists."""
    return [fn(x, y) for x, y in zip(lst1, lst2)]

def reduce(fn: Callable[[float, float], float], lst: Iterable[float], start: float) -> float:
    """Apply a function cumulatively to the items of a list, from left to right."""
    result = start
    for x in lst:
        result = fn(result, x)
    return result

# Use these to implement:
# - negList : negate a list
# - addLists : add two lists together
# - sumList: sum of a list
# - prodList: product of a list

def negList(lst: Iterable[float]) -> list:
    """Negate all elements in a list."""
    return map(neg, lst)

def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> list:
    """Element-wise addition of two lists."""
    return zipWith(add, lst1, lst2)

def sumList(lst: Iterable[float]) -> float:
    """Compute the sum of all elements in a list."""
    return reduce(add, lst, 0.0)

def prodList(lst: Iterable[float]) -> float:
    """Compute the product of all elements in a list."""
    return reduce(mul, lst, 1.0)
