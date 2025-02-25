# 1. Metaclasses
class Meta(type):
    def __new__(cls, name, bases, dct):
        print(f'Creating class {name}')
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=Meta):
    pass

# 2. Decorators
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()

# 3. Generators and Iterators
def my_generator():
    yield 1
    yield 2
    yield 3

for value in my_generator():
    print(value)

# 4. Context Managers
class MyContextManager:
    def __enter__(self):
        print("Entering the context")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting the context")

with MyContextManager():
    print("Inside the context")

# 5. Coroutines and Asyncio
import asyncio

async def my_coroutine():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

asyncio.run(my_coroutine())

# 6. Descriptor Protocol
class MyDescriptor:
    def __get__(self, instance, owner):
        return 'value'

class MyClassWithDescriptor:
    attribute = MyDescriptor()

obj = MyClassWithDescriptor()
print(obj.attribute)

# 7. Abstract Base Classes
from abc import ABC, abstractmethod

class MyAbstractClass(ABC):
    @abstractmethod
    def do_something(self):
        pass

class ConcreteClass(MyAbstractClass):
    def do_something(self):
        print("Doing something")

concrete = ConcreteClass()
concrete.do_something()

# 8. Memory Management and Garbage Collection
import gc

class MyClassForGC:
    def __del__(self):
        print("Instance is being deleted")

obj = MyClassForGC()
del obj
gc.collect()
# 9. Multithreading and Multiprocessing
import threading
import multiprocessing

def thread_function(name):
    print(f"Thread {name} starting")

def process_function(name):
    print(f"Process {name} starting")

# Multithreading
thread = threading.Thread(target=thread_function, args=(1,))
thread.start()
thread.join()

# Multiprocessing
process = multiprocessing.Process(target=process_function, args=(1,))
process.start()
process.join()

# 10. Cython and Python C API
# Note: This requires Cython to be installed and a separate .pyx file to be compiled.
# Example .pyx file content:
# def say_hello_to(name):
#     print(f"Hello {name}!")

# 11. Type Hinting and Annotations
def greeting(name: str) -> str:
    return f"Hello, {name}"

print(greeting("World"))

# 12. Data Classes
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

person = Person(name="Alice", age=30)
print(person)

# 13. Functional Programming
from functools import reduce

numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x * x, numbers))
filtered = list(filter(lambda x: x > 10, squared))
summed = reduce(lambda x, y: x + y, filtered)
print(squared, filtered, summed)

# 14. Object-Oriented Programming (OOP) Advanced Concepts
class Base:
    def __init__(self, value):
        self.value = value

    def display(self):
        print(f"Value: {self.value}")

class Derived(Base):
    def __init__(self, value, extra):
        super().__init__(value)
        self.extra = extra

    def display(self):
        super().display()
        print(f"Extra: {self.extra}")

obj = Derived(10, 20)
obj.display()

# 15. Regular Expressions
import re

pattern = r'\b\w{3}\b'
text = "The quick brown fox jumps over the lazy dog"
matches = re.findall(pattern, text)
print(matches)


# 16. Network Programming
import socket

def get_ip_address(url):
    return socket.gethostbyname(url)

print(get_ip_address('www.google.com'))

# 17. Web Scraping
import requests
from bs4 import BeautifulSoup

response = requests.get('https://www.example.com')
soup = BeautifulSoup(response.text, 'html.parser')
print(soup.title.string)

# 18. Testing and Test-Driven Development (TDD)
import unittest

def add(a, b):
    return a + b

class TestMath(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

# 19. Packaging and Distribution
# Note: This requires creating a setup.py file and using tools like setuptools.
# Example setup.py content:
# from setuptools import setup, find_packages
# setup(
#     name='mypackage',
#     version='0.1',
#     packages=find_packages(),
#     install_requires=[
#         'requests',
#         'beautifulsoup4',
#     ],
# )

# 20. Performance Optimization and Profiling
import cProfile

def slow_function():
    total = 0
    for i in range(10000):
        total += i
    return total

cProfile.run('slow_function()')










# Creating class MyClass
# Something is happening before the function is called.
# Hello!
# Something is happening after the function is called.
# 1
# 2
# 3
# Entering the context
# Inside the context
# Exiting the context
# Hello
# World
# value
# Doing something
# Instance is being deleted
# Thread 1 starting
# Process 1 starting
# Hello, World
# Person(name='Alice', age=30)
# [1, 4, 9, 16, 25] [16, 25] 41
# Value: 10
# Extra: 20
# ['The', 'fox', 'the', 'dog']
# 142.250.183.68
# Example Domain
# .
# ----------------------------------------------------------------------
# Ran 1 test in 0.000s

# OK
#          4 function calls in 0.001 seconds

#    Ordered by: standard name

#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#         1    0.000    0.000    0.001    0.001 <string>:1(<module>)
#         1    0.001    0.001    0.001    0.001 hard_code.py:218(slow_function)
#         1    0.000    0.000    0.001    0.001 {built-in method builtins.exec}
#         1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}

