# 1. List Comprehensions
squares = [x**2 for x in range(10)]  # Create a list of squares from 0 to 9
print(squares)  # Print the list of squares

# 2. Dictionary Comprehensions
squares_dict = {x: x**2 for x in range(10)}  # Create a dictionary of squares from 0 to 9
print(squares_dict)  # Print the dictionary of squares

# 3. Lambda Functions
add = lambda x, y: x + y  # Define a lambda function to add two numbers
print(add(2, 3))  # Print the result of adding 2 and 3

# 4. Map, Filter, and Reduce
from functools import reduce  # Import the reduce function from functools
nums = [1, 2, 3, 4, 5]  # Define a list of numbers
squared = list(map(lambda x: x**2, nums))  # Use map to create a list of squared numbers
even = list(filter(lambda x: x % 2 == 0, nums))  # Use filter to create a list of even numbers
sum_all = reduce(lambda x, y: x + y, nums)  # Use reduce to sum all numbers in the list
print(squared, even, sum_all)  # Print the squared numbers, even numbers, and the sum of all numbers

# 5. Decorators
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")  # Print a message before calling the function
        func()  # Call the function
        print("Something is happening after the function is called.")  # Print a message after calling the function
    return wrapper  # Return the wrapper function

@my_decorator
def say_hello():
    print("Hello!")  # Print a greeting message

say_hello()  # Call the decorated function

# 6. Generators and Iterators
def my_generator():
    for i in range(5):
        yield i  # Yield numbers from 0 to 4

gen = my_generator()  # Create a generator object
for value in gen:
    print(value)  # Print each value generated

# 7. Context Managers
with open('example.txt', 'w') as file:  # Open a file for writing
    file.write('Hello, world!')  # Write a string to the file

# 8. Exception Handling
try:
    result = 10 / 0  # Attempt to divide by zero
except ZeroDivisionError:
    print("Cannot divide by zero!")  # Print an error message if division by zero occurs

# 9. Regular Expressions
import re  # Import the regular expressions module
pattern = r'\d+'  # Define a pattern to match one or more digits
text = 'There are 123 numbers in this text.'  # Define a text string
matches = re.findall(pattern, text)  # Find all matches of the pattern in the text
print(matches)  # Print the list of matches

# 10. File I/O
with open('example.txt', 'w') as file:  # Open a file for writing
    file.write('Hello, world!')  # Write a string to the file

with open('example.txt', 'r') as file:  # Open the file for reading
    content = file.read()  # Read the content of the file
    print(content)  # Print the content of the file

# 11. JSON and XML Parsing
import json  # Import the JSON module
data = {'name': 'John', 'age': 30}  # Define a dictionary
json_str = json.dumps(data)  # Convert the dictionary to a JSON string
print(json_str)  # Print the JSON string

parsed_data = json.loads(json_str)  # Parse the JSON string back to a dictionary
print(parsed_data)  # Print the parsed dictionary

# 12. Web Scraping with BeautifulSoup
import requests  # Import the requests module
from bs4 import BeautifulSoup  # Import BeautifulSoup from bs4

response = requests.get('https://www.example.com')  # Send a GET request to a URL
soup = BeautifulSoup(response.text, 'html.parser')  # Parse the HTML content of the response
print(soup.title.string)  # Print the title of the web page

# 13. Multithreading
import threading  # Import the threading module

def print_numbers():
    for i in range(5):
        print(i)  # Print numbers from 0 to 4

thread = threading.Thread(target=print_numbers)  # Create a new thread to run the print_numbers function
thread.start()  # Start the thread
thread.join()  # Wait for the thread to finish

# 14. Multiprocessing
from multiprocessing import Process  # Import the Process class from multiprocessing

def print_numbers():
    for i in range(5):
        print(i)  # Print numbers from 0 to 4

process = Process(target=print_numbers)  # Create a new process to run the print_numbers function
process.start()  # Start the process
process.join()  # Wait for the process to finish

# 15. Asynchronous Programming (asyncio)
import asyncio  # Import the asyncio module

async def print_numbers():
    for i in range(5):
        print(i)  # Print numbers from 0 to 4
        await asyncio.sleep(1)  # Wait for 1 second

asyncio.run(print_numbers())  # Run the asynchronous function

# 16. Unit Testing (unittest, pytest)
import unittest  # Import the unittest module

def add(x, y):
    return x + y  # Define a function to add two numbers

class TestMath(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)  # Test the add function

if __name__ == '__main__':
    unittest.main()  # Run the unit tests

# 17. Logging
import logging  # Import the logging module

logging.basicConfig(level=logging.INFO)  # Configure the logging level
logging.info('This is an info message')  # Log an info message

# 18. Virtual Environments
# This is a command to be run in the terminal, not in the Python file:
# python3 -m venv myenv  # Create a virtual environment

# 19. Packaging and Distribution
# This requires creating a setup.py file, not a code snippet in the Python file:
# from setuptools import setup, find_packages
# setup(name='mypackage', version='0.1', packages=find_packages())  # Define the setup for packaging

# 20. Working with APIs (requests)
import requests  # Import the requests module

response = requests.get('https://api.github.com')  # Send a GET request to the GitHub API
print(response.json())  # Print the JSON response

# 21. Data Analysis with Pandas
import pandas as pd  # Import the pandas library

data = {'name': ['John', 'Anna'], 'age': [28, 24]}  # Define a dictionary of data
df = pd.DataFrame(data)  # Create a DataFrame from the data
print(df)  # Print the DataFrame

# 22. Data Visualization with Matplotlib and Seaborn
import matplotlib.pyplot as plt  # Import the matplotlib.pyplot module
import seaborn as sns  # Import the seaborn library

data = [1, 2, 3, 4, 5]  # Define a list of data
sns.lineplot(x=range(len(data)), y=data)  # Create a line plot using seaborn
plt.show()  # Display the plot

# 23. Object-Oriented Programming (OOP)
class Person:
    def __init__(self, name, age):
        self.name = name  # Initialize the name attribute
        self.age = age  # Initialize the age attribute

    def greet(self):
        print(f'Hello, my name is {self.name} and I am {self.age} years old.')  # Print a greeting message

person = Person('John', 30)  # Create a Person object
person.greet()  # Call the greet method

# 24. Inheritance and Polymorphism
class Animal:
    def speak(self):
        pass  # Define an abstract speak method

class Dog(Animal):
    def speak(self):
        return 'Woof!'  # Define the speak method for Dog

class Cat(Animal):
    def speak(self):
        return 'Meow!'  # Define the speak method for Cat

animals = [Dog(), Cat()]  # Create a list of Animal objects
for animal in animals:
    print(animal.speak())  # Print the sound each animal makes

# 25. Abstract Base Classes
from abc import ABC, abstractmethod  # Import ABC and abstractmethod from abc

class Animal(ABC):
    @abstractmethod
    def speak(self):
        pass  # Define an abstract speak method

class Dog(Animal):
    def speak(self):
        return 'Woof!'  # Define the speak method for Dog

dog = Dog()  # Create a Dog object
print(dog.speak())  # Print the sound the dog makes

# 26. Property Decorators (getter, setter)
class Person:
    def __init__(self, name):
        self._name = name  # Initialize the name attribute

    @property
    def name(self):
        return self._name  # Define the getter for the name attribute

    @name.setter
    def name(self, value):
        self._name = value  # Define the setter for the name attribute

person = Person('John')  # Create a Person object
print(person.name)  # Print the name attribute
person.name = 'Doe'  # Set the name attribute
print(person.name)  # Print the updated name attribute

# 27. SQLAlchemy for Database Interaction
from sqlalchemy import create_engine, Column, Integer, String, Base  # Import necessary modules from SQLAlchemy
from sqlalchemy.orm import sessionmaker  # Import sessionmaker from SQLAlchemy

engine = create_engine('sqlite:///example.db')  # Create a SQLite database engine
Base = declarative_base()  # Define the base class for declarative models

class User(Base):
    __tablename__ = 'users'  # Define the table name
    id = Column(Integer, primary_key=True)  # Define the id column
    name = Column(String)  # Define the name column

Base.metadata.create_all(engine)  # Create the table in the database
Session = sessionmaker(bind=engine)  # Create a sessionmaker
session = Session()  # Create a session

new_user = User(name='John Doe')  # Create a new User object
session.add(new_user)  # Add the new user to the session
session.commit()  # Commit the session to save the user to the database

# 28. Flask/Django for Web Development
from flask import Flask  # Import the Flask class

app = Flask(__name__)  # Create a Flask application

@app.route('/')
def home():
    return 'Hello, Flask!'  # Define a route and its handler

if __name__ == '__main__':
    app.run()  # Start the Flask web server

# 29. Socket Programming
import socket  # Import the socket module

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create a TCP/IP socket
s.bind(('localhost', 12345))  # Bind the socket to the address and port
s.listen(1)  # Listen for incoming connections
conn, addr = s.accept()  # Accept a connection
print('Connected by', addr)  # Print the address of the connected client
conn.close()  # Close the connection

# 30. Using External Libraries (e.g., NumPy, SciPy)
import numpy as np  # Import the NumPy library

array = np.array([1, 2, 3, 4, 5])  # Create a NumPy array
print(array)  # Print the array