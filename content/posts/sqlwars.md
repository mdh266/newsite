+++
authors = ["Mike Harmon"]
title = "SQL Wars: Comparing Relational Databases"
date = "2017-04-13"
tags = [
    "SQL",
    "SQLite",
    "Postgres"
]
series = ["SQL"]
aliases = ["migrate-from-jekyl"]
+++

## Contents
------------

__[1. Introduction](#Introduction)__

__[2. Preliminary Ideas](#Preliminaries)__

__[3. C.R.U.D. with SQLite](#SQLite)__

__[4. C.R.U.D. with PostgreSQL](#PostgreSQL)__

__[5. Conclusion](#Conclusion)__

--------------
## Introduction
--------------
Coming from a computational science/applied mathematics background many of the ideas of data science/machine learning are second nature to me. One thing that was very new to me was creating and manipulating databases. In computational science we don't really deal with databases and I/O is something to avoid as much as possible because it limits performance.  Therefore, in this blog post I'll be going what I have learned about SQL databases, i.e. what they are, how to set them up, and how to use them. 

In this post I'll be focusing on two SQL implementations,

* <a href="https://www.sqlite.org/">SQLite</a> and Python's interface to it <a href="https://docs.python.org/2/library/sqlite3.html">sqlite3</a>.

and 

* <a href="https://www.postgresql.org/">PostgreSQL</a> and Python's interface to it <a href="https://www.sqlalchemy.org/">sqlalchemy</a> and <a href="http://initd.org/psycopg/">psycopg2</a>.

Another option, and probably the most popular option, is <a href="https://www.mysql.com/">MySQL</a>.  However, I wont be talking about MySQL in this post. Let's first start out discussing SQL, relational databases and the client-server model for databases.

--------------
## Preliminaries
--------------

### SQL 
SQL stands for **Structured Query Language**. It is a domain specific language used in programming to deal with data that is stored in a **<a href="https://en.wikipedia.org/wiki/Relational_database">relational database</a>**.  SQL is designed for a specific purpose: to query data contained in a relational database.  There are plently of good references on how to learn SQL query commands, two that I used were,

1. <a href="http://sqlzoo.net/">SQLZOO</a>
2. <a href="https://www.w3schools.com/sql/">w3schools.com</a>

However, SQL queries are not not what I intend to cover in this post.  Instead, I would like to look at how one can create and interact with different implementations of SQL databases.  And specifically how to do this using Python.  There are many different implementations of SQL: SQLite, Oracle, MySQL, PostgreSQL, etc.  The basic operations on SQL databases that are common to all the implementations are described in the acronym, **C.R.U.D.**:

- **Create**: How to create a database and tables.

- **Read**: How read from a table in a database.

- **Update**: How to update the values in a table in the database.

- **Delete**: How to delete rows from a table in the database.

A sequence of database operations that satisfies the **A.C.I.D** properties can be perceived as single logical operation on the data, is called a **transaction.**  A.C.I.D is an acronym for set of properties relating to database transcations.  The properties are,

- **Atomicity**: The requirement that each transaction be "all or nothing." This means that if one part of the transaction fails, then the entire transaction fails, and the database state is left unchanged.


- **Consistency**: Consistency ensures that any transaction will bring the database from one valid state to another and that changes that have affected data were only completed in allowed ways. Consistency does not however, guarantee the correctness of the transcation.


- **Isolation**: This property ensures that the concurrent execution of transactions results in a system state that would be obtained if transactions were executed sequentially, i.e., one after the other


- **Durability**: Durability ensures that once a transaction has been committed, it will remain so, even in the event of power loss, crashes, or errors.


As we will see, most of the differences between working with the two implementations will be in how we create the databases.  The queries will be relatively the same, but the libaries we use to interact with the databases will be different depending on the SQL implementation. Let's now talk about the relational database model.



### Relational Database Model
This model organizes data into one or more tables (or "relations") of columns and rows, with a unique key identifying each row. Rows in a table can be linked to rows in other tables by adding a column for the unique key of the linked row (such columns are known as **foreign keys**). Most physical implementations have a unique **primary key** (PK) for each table. When a new row is written to the table, a new unique value for the primary key is generated; this is the key that the system uses primarily for accessing the table.

The primary keys within a database are used to define the relationships among the tables. When a primary key migrates to another table, it becomes a foreign key in the other table. This can be seen below where the primary key, <code>ID</code>, in the <code>School Table</code> becomes the foriegn key, <code>School ID</code>, in the <code>Student Table</code>:

<img src="https://github.com/mdh266/SQLWars/blob/master/images/Primary-Foriegn.png?raw=1">

When each cell can contain only one value and the primary key migrates into a regular entity table, this design pattern can represent either a one-to-one or one-to-many relationship. A **one-to-one relationship** exists when one row in a table may be linked with only one row in another table and vice versa. A **one-to-many relationship** exists when one row in table A may be linked with many rows in table B, but one row in table B is linked to only one row in table A. The above diagram of relationship between the <code>School Table</code> and the <code>Student Table</code> is an example of a one-to-many relationship.  There is also a **many-to-many relationship**, but this is more complicated and we won't discuss it further. The example I will work below will involve a one-to-many relationship.


### Client-Server Model For Databases

The **client-server model** is a distributed application structure that partitions tasks or workloads between the providers of a resource or service, called servers, and service requesters, called clients.  The resource/service in the case of a database is the data stored in it. Often clients and servers communicate over a computer network on separate hardware, but both client and server may reside in the same system. A server host runs one or more server programs which share their resources with clients. A client does not share any of its resources, but requests a server's content or service function. Clients therefore initiate communication sessions with servers which await incoming requests.

<img src="https://github.com/mdh266/SQLWars/blob/master/images/Client-server-model.png?raw=1">

Now that we got that basic information under our belt let's get started with SQLite first!

--------------
## SQLite
--------------
<a href="https://www.sqlite.org/">SQLite</a> was the first relational database that I learned.  It's fast, lightweight and filebased. In contrast to many other database management systems, SQLite is not a client–server database engine. Rather, it is embedded into the end program.  *SQLite stores the entire database (definitions, tables, indices, and the data itself) as a single cross-platform file on a host machine.*  SQLite is also embedded into Python naturally with <a href="https://docs.python.org/2/library/sqlite3.html">sqlite3</a> which we will be covering in this post.

Let's get started with C.R.U.D. for SQLite with Python:

### Creating A SQLite Database
First thing we need to do is import sqlite3:


```python
import sqlite3
```

Then we need to connect to our database using the command,

    conn = sqlite3.connect("database_name")
Let's do this below, calling the database <code>MyFirstDatabase.db</code>,


```python
conn = sqlite3.connect("MyFirstDatabase.db")
```

The <code>connect</code> function acts to make a "connection" to the database stored in the file <code>MyFirstDatabase.db</code>.  If the file does not exist, then it will be created automatically.  It may seem a little silly to "connect" to a file that is on our computer, but sometimes the database can reside on a different computer and the phase makes more sense.  We then use the <code>cursor()</code> function to "open" the database:


```python
cur = conn.cursor()
```

Every interaction we make with our database through python will now occur through the cursor object <code>cur</code>.  

Say, we now wish to create a table within the database called <code>Person</code> which contains the <code>name</code>, <code>id</code>, and <code>salary</code> of all employees with our company.  We note that <code>name</code> will be of type <code>TEXT</code>, which is SQLite's version of a string, while <code>id</code> and <code>salary</code> will be of type <code>INTEGER</code>.  Since each person will be uniquely identified with their <code>id</code> we make this the <code>PRIMARY KEY</code>.

The SQLite command to create this table if it doesn't already exist is,


```python
create_table = """CREATE TABLE IF NOT EXISTS Person
                (
                    name TEXT, 
                    id INTEGER PRIMARY KEY, 
                    salary INTEGER
                )"""
```

We excute this command on the SQLite database using the sqlite3 function <code>execute()</code>:


```python
cur.execute(create_table)
```




    <sqlite3.Cursor at 0x1044a6420>



**Note:** We can write SQL commands using single qoutation marks ('...'), double qoutation marks ("...") or as above with three double qoutation marks("""...""").  The latter is perferable for SQL commands for two reasons:

-  We can continue the command onto another line without the need for a backslash.


-  When using single quotation marks we would have to write a SQL string using backslashes before the single qoutations, for example:
        'name = \'Bob\' '
    This isn't necessary with double quotation marks or three double qoutation marks.

Now we'll create another table called <code>Emails</code> which contains all the employee emails.  The table will contain <code>employee_id</code> which is an <code>INTEGER</code>, <code>email</code> which is of type <code>TEXT</code> and <code>email_id</code> which is an <code>INTEGER</code>.  The <code>email_id</code> will be our <code>PRIMARY KEY</code> since it is an integer that uniquely identifies each row. The <code>FORIEGN KEY</code> in this table is then <code>employee_id</code> since it is the <code>PRIMARY KEY</code> in the <code>Person</code> table. Since employees can have more than one email address this is an example of a one-to-many relationship. The command to create the <code>Emails</code> table is then,


```python
create_table = """CREATE TABLE IF NOT EXISTS Emails 
                (
                     email_id INTEGER PRIMARY KEY,
                     employee_id INTEGER FORIEGN KEY REFERENCES Person(id), 
                     email TEXT
                 )"""
```

Notice how after we declare our foriegn key we have to say what the primary key in which table it references. Now we can execute the command:


```python
cur.execute(create_table)
```




    <sqlite3.Cursor at 0x1044a6420>



Now that we have created the tables can start to insert values into them. We use the SQL command <code>INSERT</code> to insert a new row of data into the database.  The general procedure is:

    curr.execute("""INERT INTO Table_Name 
                 (col_name_1, col_name_2, ... ) 
                 VALUES (?, ?, ... )""",
                 (col_1_value, col_2_value, ...) )
Where the tupple of column names <code>(col_name_1, col_name_2, ... )</code> of the table are followed with a <code>VALUES</code> and then followed by another tupple of '?'.  The actual column values for the row are then entered in as a separate tupple <code>(col_1_value, col_2_value, ...)</code>.  

We do this first for the <code>Person</code> table:


```python
# create dictionary of values
names = ['Bob', 'Sally', 'Wendy', 'Joe']
employee_ids = [1, 2, 3, 4]
salaries = [50, 30, 70, 20]

# loop over all the entries in the dictionary and insert them into the table
for i in range(0,4):
    cur.execute("""INSERT INTO Person 
                (name, id, salary) 
                VALUES (?, ?, ?)""",
                (names[i], employee_ids[i], salaries[i]) )

# commit the changes to disk
conn.commit()
```

The <code>commit()</code> function forces the executed SQL command to occur on the database on disk.  I'll show what happens if you don't do this in a moment, but for now let's do the same thing for the <code>Emails</code> table:


```python
# create dictionary of values
employee_ids = [1, 1, 2, 3, 4]
emails = ['bob@a.com', 'bob@b.com', 'sally@a.com', 'wendy@a.com', 'joe@a.com']
email_ids = [1, 2, 3, 4, 5]

# loop over all the entries in the dictionary and insert them into the table
for i in range(0,5):
    cur.execute("""INSERT INTO Emails 
                (email_id, employee_id, email) 
                VALUES (?, ?, ?)""",
                (email_ids[i], employee_ids[i], emails[i]) )
    
# commit the changes to disk
conn.commit()
```

### Reading From The Database
Reading from the SQLite database is rather straightforward. The following command will return a <code>sqlite.cursor</code> object:


```python
rows = cur.execute("""SELECT * FROM Person""")
print type(rows)
```

    <type 'sqlite3.Cursor'>


**Note:** If was a large database we probably dont want to read in all the rows and instead can limit to the first 100 using the command:

    SELECT * FROM Person LIMIT 100
The returned <code>sqlite.cursor</code> object will contain all the rows that were returned by the SQL query.  We can now loop over all the rows in the table and print them out with the command:


```python
for row in rows:
    print row
```

    (u'Bob', 1, 50)
    (u'Sally', 2, 30)
    (u'Wendy', 3, 70)
    (u'Joe', 4, 20)


Notice how the names of the employees are not of tpe <code>str</code>, but are rather unicode! We can also do more advanced queries like this left join:


```python
sql_querry = """SELECT name, salary, email 
              FROM Person 
              LEFT JOIN Emails
              ON (Person.id = emails.employee_id)"""

joined_rows = cur.execute(sql_querry)

# print the rows in the table
for row in joined_rows:
    print row
```

    (u'Bob', 50, u'bob@a.com')
    (u'Bob', 50, u'bob@b.com')
    (u'Sally', 30, u'sally@a.com')
    (u'Wendy', 70, u'wendy@a.com')
    (u'Joe', 20, u'joe@a.com')


### Updating Values In The Database

Updaing the values in a SQLite database is rather straight forward as well. In order to do so,  we update a row's <code>column_name1</code> value by specifying another columns value to identify that row. We'll call this other column to identify the row <code>column_name2</code>.  The general comamand is,

    UPDATE table_name SET column_name1 = val1 WHERE column_name2 = val2
Let's say Sally gets a raise and now her income changes from 30 to 40, we can do this with the SQLite command,


```python
# Update Sally's salary
cur.execute("""UPDATE Person SET salary = 40 WHERE name = 'Sally' """)

# read in the table and print out the rows
rows = cur.execute("""SELECT * FROM Person""")
for row in rows:
    print row
```

    (u'Bob', 1, 50)
    (u'Sally', 2, 40)
    (u'Wendy', 3, 70)
    (u'Joe', 4, 20)


But this only changes the local version of the table and does not persist to the database on disk.  We can observe this firsthand if we close the connection to the database, then reconnect again and print out the values:


```python
# close the connection and reopen it.
conn.close()
conn = sqlite3.connect("MyFirstDatabase.db")
cur = conn.cursor()

# read in the table and print out the rows
rows = cur.execute("""SELECT * FROM Person""")
for row in rows:
    print row
```

    (u'Bob', 1, 50)
    (u'Sally', 2, 30)
    (u'Wendy', 3, 70)
    (u'Joe', 4, 20)


Now we will commit the update to the database on disk and close the connection. Then we will re-connect to it and print out the data to make sure that it has changed:


```python
# update the row in the table and commit the change to disk
cur.execute("""UPDATE Person SET salary = 40 WHERE name = 'Sally' """)
conn.commit()
conn.close()

# reopen the connection and print the rows in the table
conn = sqlite3.connect("MyFirstDatabase.db")
cur = conn.cursor()
rows = cur.execute("""SELECT * FROM Person""")
for row in rows:
    print row
```

    (u'Bob', 1, 50)
    (u'Sally', 2, 40)
    (u'Wendy', 3, 70)
    (u'Joe', 4, 20)


Now we can see that database has been updated!

### Deleting Rows From The Database

To delete a row from a table in a SQLite database we have to identify the row by one of its column values.  The general command is,

    DELETE FROM table_name WHERE column_name = val
We can delete Bob from the <code>Person</code> table with the following command:


```python
# delete the row
cur.execute("""DELETE FROM Person WHERE name = 'Bob' """)

# print the rows in the table
rows = cur.execute("""SELECT * FROM Person""")
for row in rows:
    print row
```

    (u'Sally', 2, 40)
    (u'Wendy', 3, 70)
    (u'Joe', 4, 20)


Now that we have covered the basic C.R.U.D commands in SQLite let's move on to PostgreSQL!

--------------
## PostgreSQL
--------------

While SQLite is fast and easy, it can't handle many concurrent writes. It is also not great for large databases since everything must fit in one file on disk.  PostgreSQL is another great option for a relation database that is powerful and open source.  PostgreSQL databases are not limited in size and can handle multiple concurrent writes.  PostgreSQL manages concurrency through a system known as **multiversion concurrency control (MVCC)**, which gives each transaction a "snapshot" of the database, allowing changes to be made without being visible to other transactions until the changes are committed. This largely eliminates the need for read locks, and ensures the database maintains the ACID (atomicity, consistency, isolation, durability) principles in an efficient manner. The one draw back to PostgreSQL is that is a little harder to set up, which is what I will be focusing on below.

### Creating A Database

The first thing we will need to do is import the libraries to interact with SQL in Python.  These are <a href="https://www.sqlalchemy.org/">sqlalchemy</a> and <a href="https://sqlalchemy-utils.readthedocs.io/en/latest/">sqlalchemy_utils</a>. We import them in below,


```python
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
```

    dialect+driver://username:password@host:port/database

Above we see the general command for how to create a SQL database using Python.  The dialect in our case will be <code>postgresql</code> and the driver will be <code>psycopg2</code>. We import the <a href="http://initd.org/psycopg/">psycopg2</a> library here,


```python
import psycopg2
```

Now we define our user name, <code>username</code>, and database name, <code>dbname</code>:


```python
dbname = 'AnotherDatabase'
username = 'Mike'
```

Then we create the database engine using the above command with the values filled in. Notice below how we do not include the password since we don't want to use one.  We also make the host to be <code>localhost</code> since we are going to have the database local on our machine.


```python
engine = create_engine('postgresql+psycopg2://%s@localhost/%s'%(username,dbname))
print engine.url
```

    postgresql+psycopg2://Mike@localhost/AnotherDatabase


Now create the database, if it doesn't already exist:


```python
if not database_exists(engine.url):
    create_database(engine.url)
print(database_exists(engine.url))
```

    True


We connect to our database, <code>dbname</code> with the psycopg2 library using the command :
    
    psycopg2.connect(database = dbname, user = username, password = user_password)
where <code>user_password</code> is our password. To connect to our newly created PostgreSQL database we then type,


```python
conn = psycopg2.connect(database = 'AnotherDatabase', user = 'Mike')
```

We now get our cursor object:


```python
cur = conn.cursor()
```

And we can create the same <code>Person</code> table as before, except this time instead of <code>name</code> being of type <code>TEXT</code> it is of type <code>VARCHAR</code>.


```python
create_table = """CREATE TABLE IF NOT EXISTS Person 
                (
                    name VARCHAR, 
                    id INTEGER PRIMARY KEY, 
                    salary INTEGER
                )
                """
cur.execute(create_table)
```

Inserting the data with PostgresSQL is almost exactly the same as with SQLite except instead of the command using '?' in the tupple after <code>VALUES</code> we use '%s'.  

We fill in the values for the <code>Person</code> table below:


```python
names = ['Bob', 'Sally', 'Wendy', 'Joe']
employee_ids = [1, 2, 3, 4]
salaries = [50, 30, 70, 20]

for i in range(0,4):
    cur.execute("""INSERT INTO Person 
                (name, id, salary) 
                VALUES (%s, %s, %s);""",
                (names[i], employee_ids[i], salaries[i]) )

conn.commit()
```

We repeat the same process for the <code>Email</code> table, except with PostgreSQL we don't have to explicitly state that <code>employee_id</code> is the <code>FOREIGN KEY</code>, but we do still have to say what <code>PRIMARY KEY</code> it references:


```python
# create the table
create_table = """CREATE TABLE IF NOT EXISTS Emails 
                (
                     email_id INTEGER PRIMARY KEY,
                     employee_id INTEGER REFERENCES Person(id), 
                     email VARCHAR
                 )"""

cur.execute(create_table)

# fill in the table
employee_ids = [1, 1, 2, 3, 4]
email_ids = [1, 2, 3, 4, 5]
emails = ['bob@a.com', 'bob@b.com', 'sally@a.com', 'wendy@a.com', 'joe@a.com']

for i in range(0,5):
    cur.execute("""INSERT INTO Emails 
                (email_id, employee_id, email) 
                VALUES (%s, %s, %s)""",
                (email_ids[i], employee_ids[i], emails[i]) )

conn.commit()
conn.close()
```

### Reading From The Database

Now we want to read data in from the table in the database.  We first have to connect with the database again and then set the cursor object:


```python
conn = psycopg2.connect(database = dbname, user = username)
cur = conn.cursor()
```

Now the command to select the rows from the table are exactly the same as with SQLite.


```python
cur.execute("""SELECT * FROM Person""")
```

However, now the way we view the rows in the table using PostgreSQL is different from SQLite.  The command above does not return an object of the rows as it did with sqlite3.  Instead, now we get the rows in the table by using one of the following commands on cur:

    cur.fetchone() 
    
    cur.fetchmany() 
        
    cur.fetchall()
        
For information on these commands see <a href="http://stackoverflow.com/questions/5189997/python-db-api-fetchone-vs-fetchmany-vs-fetchall">this</a> Stack Overflow post. Since our table is so small we'll fetch all the rows:


```python
cur.fetchall()
```




    [('Bob', 1, 50), ('Sally', 2, 30), ('Wendy', 3, 70), ('Joe', 4, 20)]



We can also do the same left join as before:


```python
sql_querry = """SELECT name, salary, email 
              FROM Person 
              LEFT JOIN Emails
              ON (Person.id = emails.employee_id)"""

cur.execute(sql_querry)
cur.fetchall()
```




    [('Bob', 50, 'bob@a.com'),
     ('Bob', 50, 'bob@b.com'),
     ('Sally', 30, 'sally@a.com'),
     ('Wendy', 70, 'wendy@a.com'),
     ('Joe', 20, 'joe@a.com')]



### Updating And Deleting Rows In The Database

Updating values and deleting rows in PostgreSQL is exactly the same as in SQLite and don't go into here.


```python
conn.close()
```

--------------
## Conclusion
--------------
I'll be upating this more as time goes on, but for now we can summarize some of the highlights of SQLite and PostgreSQL are:

### SQLite
SQLite Is probably best for mobile apps or personal projects or where the dataset is not too big and can be stored locally on the machine.

**PROS:** Requires less configuration than client-server based database. Easy to use, fast, lightweight and has native bindings to Python.

**CONS:** Cannot handle concurrent writes, instead writes must be performed sequentially.  SQLite can handle concurrent reads, but not too many.


### PostgreSQL

PostgreSQL is the best bet for a large scale open-source databases that will have heavy user traffic.

**PROS:** Advanced open-source database that scales to large datasets. Can handle multiple concurrent write and reads.

**CONS:** Can be more complex to set up.
