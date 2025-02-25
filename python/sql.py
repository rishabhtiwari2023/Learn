# GitHub Copilot
# Selecting Data
# Filtering Data
# Sorting Data
# Joining Tables
# Grouping Data
# Aggregating Data
# Subqueries
# Common Table Expressions (CTEs)
# Window Functions
# Inserting Data
# Updating Data
# Deleting Data
# Creating Tables
# Altering Tables
# Dropping Tables
# Creating Indexes
# Using Indexes
# Transactions
# Views
# Stored Procedures
# Triggers
# User-Defined Functions
# Constraints
# Foreign Keys
# Primary Keys
# Unique Constraints
# Check Constraints
# Default Values
# Data Types0
# Backup and Restore


import sqlite3

# Connect to SQLite database
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Create a sample table
cursor.execute('''
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER
)
''')

# Insert sample data
cursor.executemany('''
INSERT INTO users (name, age) VALUES (?, ?)
''', [('Alice', 30), ('Bob', 25), ('Charlie', 35)])

# 1. Selecting Data
cursor.execute('SELECT * FROM users')
print("Selecting Data:", cursor.fetchall())

# 2. Filtering Data
cursor.execute('SELECT * FROM users WHERE age > 30')
print("Filtering Data:", cursor.fetchall())

# 3. Sorting Data
cursor.execute('SELECT * FROM users ORDER BY age DESC')
print("Sorting Data:", cursor.fetchall())

# 4. Joining Tables
cursor.execute('''
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    amount REAL,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
''')
cursor.executemany('''
INSERT INTO orders (user_id, amount) VALUES (?, ?)
''', [(1, 100.0), (2, 200.0), (1, 150.0)])

cursor.execute('''
SELECT users.name, orders.amount 
FROM users 
JOIN orders ON users.id = orders.user_id
''')
print("Joining Tables:", cursor.fetchall())

# 5. Grouping Data
cursor.execute('''
SELECT user_id, COUNT(*) as order_count 
FROM orders 
GROUP BY user_id
''')
print("Grouping Data:", cursor.fetchall())

# 6. Aggregating Data
cursor.execute('''
SELECT user_id, SUM(amount) as total_amount 
FROM orders 
GROUP BY user_id
''')
print("Aggregating Data:", cursor.fetchall())

# 7. Subqueries
cursor.execute('''
SELECT name 
FROM users 
WHERE id IN (SELECT user_id FROM orders WHERE amount > 100)
''')
print("Subqueries:", cursor.fetchall())

# 8. Common Table Expressions (CTEs)
cursor.execute('''
WITH OrderTotals AS (
    SELECT user_id, SUM(amount) as total_amount 
    FROM orders 
    GROUP BY user_id
)
SELECT users.name, OrderTotals.total_amount 
FROM users 
JOIN OrderTotals ON users.id = OrderTotals.user_id
''')
print("CTEs:", cursor.fetchall())

# Close the connection
conn.close()

import sqlite3

# Connect to SQLite database
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Create a sample table
cursor.execute('''
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER
)
''')

# Insert sample data
cursor.executemany('''
INSERT INTO users (name, age) VALUES (?, ?)
''', [('Alice', 30), ('Bob', 25), ('Charlie', 35)])

# Create orders table
cursor.execute('''
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    amount REAL,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
''')
cursor.executemany('''
INSERT INTO orders (user_id, amount) VALUES (?, ?)
''', [(1, 100.0), (2, 200.0), (1, 150.0)])

# 9. Window Functions
cursor.execute('''
SELECT name, age, 
       ROW_NUMBER() OVER (ORDER BY age) as row_num
FROM users
''')
print("Window Functions:", cursor.fetchall())

# 10. Inserting Data
cursor.execute('''
INSERT INTO users (name, age) VALUES ('David', 40)
''')
conn.commit()
cursor.execute('SELECT * FROM users WHERE name = "David"')
print("Inserting Data:", cursor.fetchall())

# 11. Updating Data
cursor.execute('''
UPDATE users SET age = 45 WHERE name = 'David'
''')
conn.commit()
cursor.execute('SELECT * FROM users WHERE name = "David"')
print("Updating Data:", cursor.fetchall())

# 12. Deleting Data
cursor.execute('''
DELETE FROM users WHERE name = 'David'
''')
conn.commit()
cursor.execute('SELECT * FROM users WHERE name = "David"')
print("Deleting Data:", cursor.fetchall())

# 13. Creating Tables
cursor.execute('''
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT,
    price REAL
)
''')
print("Creating Tables: Table 'products' created")

# 14. Altering Tables
cursor.execute('''
ALTER TABLE products ADD COLUMN stock INTEGER DEFAULT 0
''')
print("Altering Tables: Column 'stock' added to 'products'")

# 15. Dropping Tables
cursor.execute('''
DROP TABLE products
''')
print("Dropping Tables: Table 'products' dropped")

# 16. Creating Indexes
cursor.execute('''
CREATE INDEX idx_user_age ON users(age)
''')
print("Creating Indexes: Index 'idx_user_age' created on 'users' table")

# Close the connection
conn.close()



import sqlite3

# Connect to SQLite database
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Create a sample table
cursor.execute('''
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER
)
''')

# Insert sample data
cursor.executemany('''
INSERT INTO users (name, age) VALUES (?, ?)
''', [('Alice', 30), ('Bob', 25), ('Charlie', 35)])

# Create orders table
cursor.execute('''
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    amount REAL,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
''')
cursor.executemany('''
INSERT INTO orders (user_id, amount) VALUES (?, ?)
''', [(1, 100.0), (2, 200.0), (1, 150.0)])

# 17. Using Indexes
cursor.execute('''
EXPLAIN QUERY PLAN SELECT * FROM users WHERE age = 30
''')
print("Using Indexes:", cursor.fetchall())

# 18. Transactions
try:
    conn.execute('BEGIN TRANSACTION')
    cursor.execute('''
    INSERT INTO users (name, age) VALUES ('David', 40)
    ''')
    cursor.execute('''
    UPDATE users SET age = 45 WHERE name = 'David'
    ''')
    conn.commit()
    print("Transactions: Transaction committed")
except:
    conn.rollback()
    print("Transactions: Transaction rolled back")

# 19. Views
cursor.execute('''
CREATE VIEW user_orders AS
SELECT users.name, orders.amount 
FROM users 
JOIN orders ON users.id = orders.user_id
''')
cursor.execute('SELECT * FROM user_orders')
print("Views:", cursor.fetchall())

# 20. Stored Procedures
# SQLite does not support stored procedures directly, but we can simulate it using functions
def add_user(name, age):
    cursor.execute('''
    INSERT INTO users (name, age) VALUES (?, ?)
    ''', (name, age))
    conn.commit()

add_user('Eve', 28)
cursor.execute('SELECT * FROM users WHERE name = "Eve"')
print("Stored Procedures:", cursor.fetchall())

# 21. Triggers
cursor.execute('''
CREATE TRIGGER user_update_trigger
AFTER UPDATE ON users
BEGIN
    INSERT INTO orders (user_id, amount) VALUES (NEW.id, 0);
END;
''')
cursor.execute('''
UPDATE users SET age = 50 WHERE name = 'Eve'
''')
cursor.execute('SELECT * FROM orders WHERE user_id = (SELECT id FROM users WHERE name = "Eve")')
print("Triggers:", cursor.fetchall())

# 22. User-Defined Functions
def calculate_discount(amount):
    return amount * 0.9

conn.create_function("discount", 1, calculate_discount)
cursor.execute('''
SELECT amount, discount(amount) as discounted_amount FROM orders
''')
print("User-Defined Functions:", cursor.fetchall())

# 23. Constraints
cursor.execute('''
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    price REAL CHECK(price > 0)
)
''')
try:
    cursor.execute('''
    INSERT INTO products (name, price) VALUES ('Product1', -10)
    ''')
except sqlite3.IntegrityError as e:
    print("Constraints: IntegrityError -", e)

# 24. Foreign Keys
cursor.execute('''
CREATE TABLE categories (
    id INTEGER PRIMARY KEY,
    name TEXT
)
''')
cursor.execute('''
CREATE TABLE product_categories (
    product_id INTEGER,
    category_id INTEGER,
    FOREIGN KEY(product_id) REFERENCES products(id),
    FOREIGN KEY(category_id) REFERENCES categories(id)
)
''')
print("Foreign Keys: Tables 'categories' and 'product_categories' created")

# Close the connection
conn.close()




import sqlite3

# Connect to SQLite database
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Create a sample table
cursor.execute('''
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER
)
''')

# Insert sample data
cursor.executemany('''
INSERT INTO users (name, age) VALUES (?, ?)
''', [('Alice', 30), ('Bob', 25), ('Charlie', 35)])

# Create orders table
cursor.execute('''
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    amount REAL,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
''')
cursor.executemany('''
INSERT INTO orders (user_id, amount) VALUES (?, ?)
''', [(1, 100.0), (2, 200.0), (1, 150.0)])

# 24. Foreign Keys
cursor.execute('''
CREATE TABLE categories (
    id INTEGER PRIMARY KEY,
    name TEXT
)
''')
cursor.execute('''
CREATE TABLE product_categories (
    product_id INTEGER,
    category_id INTEGER,
    FOREIGN KEY(product_id) REFERENCES products(id),
    FOREIGN KEY(category_id) REFERENCES categories(id)
)
''')
print("Foreign Keys: Tables 'categories' and 'product_categories' created")

# 25. Primary Keys
cursor.execute('''
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    position TEXT
)
''')
print("Primary Keys: Table 'employees' created with primary key 'id'")

# 26. Unique Constraints
cursor.execute('''
CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
)
''')
try:
    cursor.execute('''
    INSERT INTO departments (name) VALUES ('HR'), ('HR')
    ''')
except sqlite3.IntegrityError as e:
    print("Unique Constraints: IntegrityError -", e)

# 27. Check Constraints
cursor.execute('''
CREATE TABLE salaries (
    id INTEGER PRIMARY KEY,
    amount REAL CHECK(amount > 0)
)
''')
try:
    cursor.execute('''
    INSERT INTO salaries (amount) VALUES (-100)
    ''')
except sqlite3.IntegrityError as e:
    print("Check Constraints: IntegrityError -", e)

# 28. Default Values
cursor.execute('''
CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name TEXT,
    start_date TEXT DEFAULT (date('now'))
)
''')
cursor.execute('''
INSERT INTO projects (name) VALUES ('Project1')
''')
cursor.execute('SELECT * FROM projects')
print("Default Values:", cursor.fetchall())

# 29. Data Types
cursor.execute('''
CREATE TABLE assets (
    id INTEGER PRIMARY KEY,
    name TEXT,
    value REAL
)
''')
cursor.execute('''
INSERT INTO assets (name, value) VALUES ('Asset1', 1000.50)
''')
cursor.execute('SELECT * FROM assets')
print("Data Types:", cursor.fetchall())

# 30. Backup and Restore
# Backup
backup_conn = sqlite3.connect('backup.db')
with backup_conn:
    conn.backup(backup_conn)
print("Backup: Database backed up to 'backup.db'")

# Restore
restore_conn = sqlite3.connect(':memory:')
with restore_conn:
    backup_conn.backup(restore_conn)
restore_cursor = restore_conn.cursor()
restore_cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
print("Restore: Tables in restored database:", restore_cursor.fetchall())

# Close the connections
conn.close()
backup_conn.close()
restore_conn.close()