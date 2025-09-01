+++
authors = ["Mike Harmon"]
title = "Polars & DuckDB: DataFrames and SQL Without Pandas On AWS"
date = "2023-07-19"
tags = [
    "Polars",
    "DuckDB",
    "SQL",
    "AWS",
    "Docker"
]
series = ["SQL"]
aliases = ["migrate-from-jekyl"]
+++


## Contents
--------------------------

__[1. Introduction](#first-bullet)__

__[2. Getting Set Up On AWS with Docker](#second-bullet)__

__[3. Intro To Polars](#third-bullet)__

__[4. DuckDB To The Rescue For SQL](#fourth-bullet)__

__[5. Conclusions](#fifth)__


## Introduction <a class="anchor" id="first-bullet"></a>
------

In the last few years there has been an explosion of dataframe alternatives to [Pandas](https://pandas.pydata.org/) due to its [limitations](https://insightsndata.com/what-are-the-limitations-of-pandas-35d462990c43). Even the original author, Wes McKinney, wrote a blog post about [10 Things I Hate About Pandas](https://wesmckinney.com/blog/apache-arrow-pandas-internals/). 

My biggest complaints about Pandas are:

1. High memory usage
2. Limited multi-core algorithms
3. No ability to execute SQL statements (like [SparkSQL & DataFrame](https://spark.apache.org/sql/))
4. No query planning/lazy-execution
5. [NULL values only exist for floats not ints](https://pandas.pydata.org/docs/user_guide/integer_na.html) (this changed in Pandas 1.0+)
6. Using [strings is inefficient](https://pandas.pydata.org/docs/user_guide/text.html) (this too changed in Pandas 1.0+
    
I should note that many of these issues have been addressed by the [Pandas 2.0 release](https://pandas.pydata.org/docs/dev/whatsnew/v2.0.0.html). And while there has been a steady march towards replacing the [NumPy](https://numpy.org/) backend with [Apache Arrow](https://arrow.apache.org/), I still feel the lack of SQL and overall API design is a major weakness of Pandas. Let me expand upon tha last point.

For context I have been using a [Apache Spark](https://spark.apache.org/) since 2017 and love it not just from a performance point of view, but I also love how well the API is designed. The syntax makes sense coming from a SQL users perspective. If I want to group by a column and count in SQL or on Spark DataFrame I get what I expect either way: *A single column with the count of each item the original dataframes/tables column.* In Pandas, this is not the result.

For example using this datas set from [NYC Open Data](https://opendata.cityofnewyork.us/) on [Motor Vechicle Collisions](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95), I can run a groupby-count expression on a Pandas DataFrame and I get:


```python
import pandas as pd
pd_df = pd.read_csv("https://data.cityofnewyork.us/resource/h9gi-nx95.csv")
pd_df.groupby("borough").count()
```




<div style="overflow-x: auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>crash_date</th>
      <th>crash_time</th>
      <th>zip_code</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>location</th>
      <th>on_street_name</th>
      <th>off_street_name</th>
      <th>cross_street_name</th>
      <th>number_of_persons_injured</th>
      <th>...</th>
      <th>contributing_factor_vehicle_2</th>
      <th>contributing_factor_vehicle_3</th>
      <th>contributing_factor_vehicle_4</th>
      <th>contributing_factor_vehicle_5</th>
      <th>collision_id</th>
      <th>vehicle_type_code1</th>
      <th>vehicle_type_code2</th>
      <th>vehicle_type_code_3</th>
      <th>vehicle_type_code_4</th>
      <th>vehicle_type_code_5</th>
    </tr>
    <tr>
      <th>borough</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BRONX</th>
      <td>107</td>
      <td>107</td>
      <td>107</td>
      <td>107</td>
      <td>107</td>
      <td>107</td>
      <td>59</td>
      <td>59</td>
      <td>48</td>
      <td>107</td>
      <td>...</td>
      <td>81</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>107</td>
      <td>106</td>
      <td>65</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BROOKLYN</th>
      <td>247</td>
      <td>247</td>
      <td>247</td>
      <td>245</td>
      <td>245</td>
      <td>245</td>
      <td>155</td>
      <td>155</td>
      <td>92</td>
      <td>247</td>
      <td>...</td>
      <td>192</td>
      <td>24</td>
      <td>7</td>
      <td>2</td>
      <td>247</td>
      <td>242</td>
      <td>157</td>
      <td>22</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>MANHATTAN</th>
      <td>98</td>
      <td>98</td>
      <td>98</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>52</td>
      <td>52</td>
      <td>46</td>
      <td>98</td>
      <td>...</td>
      <td>65</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>98</td>
      <td>96</td>
      <td>57</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>QUEENS</th>
      <td>154</td>
      <td>154</td>
      <td>153</td>
      <td>150</td>
      <td>150</td>
      <td>150</td>
      <td>98</td>
      <td>98</td>
      <td>56</td>
      <td>154</td>
      <td>...</td>
      <td>120</td>
      <td>9</td>
      <td>2</td>
      <td>0</td>
      <td>154</td>
      <td>154</td>
      <td>97</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>STATEN ISLAND</th>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>26</td>
      <td>26</td>
      <td>26</td>
      <td>18</td>
      <td>18</td>
      <td>9</td>
      <td>27</td>
      <td>...</td>
      <td>21</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>27</td>
      <td>27</td>
      <td>19</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



Notice this is the number of non nulls in every column. Not exactly what I wanted.

To get what I want I have to use the syntax:


```python
pd_df.groupby("borough").size() # or pd_df.value_counts()
```




    borough
    BRONX            107
    BROOKLYN         247
    MANHATTAN         98
    QUEENS           154
    STATEN ISLAND     27
    dtype: int64



But this returns a [Pandas Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html). It seems like a trivial difference, but counting duplicates in a column is easy in Spark because we can use method chaining, to the do the equivalent in Pandas I have to convert the series back to a dataframe and reset the index first:


```python
pd_df.groupby("borough").size().to_frame("counts").reset_index().query("counts > 0")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>borough</th>
      <th>counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BRONX</td>
      <td>107</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BROOKLYN</td>
      <td>247</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MANHATTAN</td>
      <td>98</td>
    </tr>
    <tr>
      <th>3</th>
      <td>QUEENS</td>
      <td>154</td>
    </tr>
    <tr>
      <th>4</th>
      <td>STATEN ISLAND</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>



Furthermore, **in Pandas there are too many ways to do the same thing.**  In my opinion, in a well designed API this shouldn't be the case. Lastly, in Pandas, window functions, which are incredibly import in SQL are just awkward to write.

For years I have been using Spark for large datasets, but for smaller ones sticking with Pandas and making do. Recently though, I heard lots of hype about [Polars](https://www.pola.rs/) and [DuckDB](https://duckdb.org/) and decide to try them myself and was immediately impressed. In my opinion, Polars is not 100% mature yet, but I still  has a lot of potential, many because for me the API is much more similar to Spark's than Pandas is.

In this blog post I go over my first interactions with both libraries and call out things I like and do not like, but first let's get set up to run this notebook on an AWS EC2 instance using [Docker](https://www.docker.com/).

## Getting Set Up On AWS with Docker <a class="anchor" id="second-bullet"></a>

I have mostly used [Google Cloud](https://cloud.google.com/) for my prior personal projects, but for this project I wanted to use [Amazon Web Services](https://aws.com/). The first thing I do is create a [S3 bucket](https://aws.amazon.com/s3/). I do this from the console by signing on to [aws.com](aws.com) and going to the S3 page:


<img src="https://github.com/mdh266/PolarsDuckDBPlayGround/blob/main/images/s3.png?raw=1" width="1000">

I can click the `Create bucket` button and create a bucket called `harmonskis` (for funskis) with all the default settings and click the`Create bucket` button on the bottom right side.

Next I need to have access to read and write to and from the S3 bucket so I create an [IAM role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html) to do so. Going to the signin dashboard I can search for "IAM" and click on the link. This takes me to another site where selecting the "Roles" link in the the "Access Management" drop down on the left hand side takes me to the following:

<img src="https://github.com/mdh266/PolarsDuckDBPlayGround/blob/main/images/IAM.png?raw=1" width="1000">

I can click create the `Create role` button on the top right that takes me to the page:

<img src="https://github.com/mdh266/PolarsDuckDBPlayGround/blob/main/images/ec2-role.png?raw=1" width="1000">

I keep the selection of "AWS Service", select the "ec2" option and then click the `Next` button on the bottom right. This takes me to a page where I can create a policy. Searching for "s3" I select the following policy that gives me read/write access:

<img src="https://github.com/mdh266/PolarsDuckDBPlayGround/blob/main/images/create_policy.png?raw=1" width="1000">

I then click the `Next` button in the bottom right which takes me to the final page:

<img src="https://github.com/mdh266/PolarsDuckDBPlayGround/blob/main/images/role.png?raw=1" width="1000">

I give the role the name "s3acess" (spelling isnt my best skill) and then click `Create role` in the bottom right.

Next I will create my [Elastic Compute Cloud 
(EC2) Instance](https://aws.amazon.com/ec2/) instance by going to the console again and clicking on ec2, scrolling down and clicking the orange `Launch instance` button,

<img src="https://github.com/mdh266/PolarsDuckDBPlayGround/blob/main/images/launch.png?raw=1" width="1000">

Next I have to make sure I create a `keypair` file called "mikeskey.pem" that I download.

<img src="https://github.com/mdh266/PolarsDuckDBPlayGround/blob/main/images/keypair.png?raw=1" width="1000">

Notice that in the security group I use allows SSH traffic from "Anywhere". Finally, under the "Advanced details" drop down I select "s3acess" (I'm living with my spelling mistake) from the "IAM instance policy":

<img src="https://github.com/mdh266/PolarsDuckDBPlayGround/blob/main/images/s3access.png?raw=1" width="1000">

Once I launch the EC2 instance I can see the instance running and click on `Instance ID` as shown below:

<img src="https://github.com/mdh266/PolarsDuckDBPlayGround/blob/main/images/instance.png?raw=1" width="1000">


I can then click on the pop up choice of `Connect`. This takes me to another page where I get the command at the bottom of the page to SSH onto my machine using the keypair I created:

<img src="https://github.com/mdh266/PolarsDuckDBPlayGround/blob/main/images/connect.png?raw=1" width="1000">


I could ssh onto the server with the following command:

    ssh -i <path-to-key>/mikeskey.pem ec2-user@<dns-address>.compute-1.amazonaws.com

Note that I didnt create a user name so it defaulted to `ec2-user`. 

However, since I'll be running jupyter lab on a remote EC2 server I need to set up [ssh-tunneling](https://linuxize.com/post/how-to-setup-ssh-tunneling/) as described [here](https://towardsdatascience.com/setting-up-and-using-jupyter-notebooks-on-aws-61a9648db6c5) so that I can access it from the web browser on my laptop. I can do this by running the command:

    ssh -i <path-to-key>/mikeskey.pem -L 8888:localhost:8888 ec2-user@<dns-address>.compute-1.amazonaws.com

Next I set up git ssh-keys so I could develop on the instance as described [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) and clone the repo. I can then set up Docker as discussed [here](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/create-container-image.html). Then I build the image and call it `polars_nb`:

    sudo docker build -t polars_nb . 

Finally, I start up the container from this image using port forwarding and loading the current directory as the volume:

    sudo docker run -ip 8888:8888 -v `pwd`:/home/jovyan/ -t polars_nb

The terminal shows a link that I can copy and paste into my webbrowser, I make sure to copy the one with the 127 in it and viola it works!

## Intro To Polars <a class="anchor" id="third-bullet"></a>

Now that we're set up with a notebook on an EC2 isntance we can start to discuss [Polars](https://www.pola.rs/) dataframes. The Polars library is written in Rust with Python bindings. Polars uses multi-core processing making it fast and the authors smartly used [Apache Arrow](https://arrow.apache.org/) making it efficient for cross-language in-memory dataframes as there is no serialization between the Rust and Python. According to the website the philosophy of Polars is,

The goal of Polars is to provide a lightning fast DataFrame library that:

* Utilizes all available cores on your machine.
* Optimizes queries to reduce unneeded work/memory allocations.
* Handles datasets much larger than your available RAM.
* Has an API that is consistent and predictable.
* Has a strict schema (data-types should be known before running the query).

Let's get started! We can import polars and read in a dataset from [NY Open Data](https://opendata.cityofnewyork.us/) on [Motor Vehicle Collisions](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) using the [read_csv](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.read_csv.html) function:


```python
import polars as pl
df = pl.read_csv("https://data.cityofnewyork.us/resource/h9gi-nx95.csv")
df.head(2)
```





<div style="overflow-x: auto;">
<style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (2, 29)</small><table border="1" class="dataframe"><thead><tr><th>crash_date</th><th>crash_time</th><th>borough</th><th>zip_code</th><th>latitude</th><th>longitude</th><th>location</th><th>on_street_name</th><th>off_street_name</th><th>cross_street_name</th><th>number_of_persons_injured</th><th>number_of_persons_killed</th><th>number_of_pedestrians_injured</th><th>number_of_pedestrians_killed</th><th>number_of_cyclist_injured</th><th>number_of_cyclist_killed</th><th>number_of_motorist_injured</th><th>number_of_motorist_killed</th><th>contributing_factor_vehicle_1</th><th>contributing_factor_vehicle_2</th><th>contributing_factor_vehicle_3</th><th>contributing_factor_vehicle_4</th><th>contributing_factor_vehicle_5</th><th>collision_id</th><th>vehicle_type_code1</th><th>vehicle_type_code2</th><th>vehicle_type_code_3</th><th>vehicle_type_code_4</th><th>vehicle_type_code_5</th></tr><tr><td>str</td><td>str</td><td>str</td><td>i64</td><td>f64</td><td>f64</td><td>str</td><td>str</td><td>str</td><td>str</td><td>i64</td><td>i64</td><td>i64</td><td>i64</td><td>i64</td><td>i64</td><td>i64</td><td>i64</td><td>str</td><td>str</td><td>str</td><td>str</td><td>str</td><td>i64</td><td>str</td><td>str</td><td>str</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>&quot;2021-09-11T00:…</td><td>&quot;2:39&quot;</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>&quot;WHITESTONE EXP…</td><td>&quot;20 AVENUE&quot;</td><td>null</td><td>2</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2</td><td>0</td><td>&quot;Aggressive Dri…</td><td>&quot;Unspecified&quot;</td><td>null</td><td>null</td><td>null</td><td>4455765</td><td>&quot;Sedan&quot;</td><td>&quot;Sedan&quot;</td><td>null</td><td>null</td><td>null</td></tr><tr><td>&quot;2022-03-26T00:…</td><td>&quot;11:45&quot;</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>&quot;QUEENSBORO BRI…</td><td>null</td><td>null</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>&quot;Pavement Slipp…</td><td>null</td><td>null</td><td>null</td><td>null</td><td>4513547</td><td>&quot;Sedan&quot;</td><td>null</td><td>null</td><td>null</td><td>null</td></tr></tbody></table></div>



The initial reading of CSVs is the same as Pandas and the [head](https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.head.html) dataframe method returns the top `n` rows as Pandas does. However, in addition to the printed rows, I also get shape of the dataframe as well as the datatypes of the columns. 

I can get the name of columns and their datatypes using the [schema](https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.schema.html) method which is similar to Spark:


```python
df.schema
```




    {'crash_date': Utf8,
     'crash_time': Utf8,
     'borough': Utf8,
     'zip_code': Int64,
     'latitude': Float64,
     'longitude': Float64,
     'location': Utf8,
     'on_street_name': Utf8,
     'off_street_name': Utf8,
     'cross_street_name': Utf8,
     'number_of_persons_injured': Int64,
     'number_of_persons_killed': Int64,
     'number_of_pedestrians_injured': Int64,
     'number_of_pedestrians_killed': Int64,
     'number_of_cyclist_injured': Int64,
     'number_of_cyclist_killed': Int64,
     'number_of_motorist_injured': Int64,
     'number_of_motorist_killed': Int64,
     'contributing_factor_vehicle_1': Utf8,
     'contributing_factor_vehicle_2': Utf8,
     'contributing_factor_vehicle_3': Utf8,
     'contributing_factor_vehicle_4': Utf8,
     'contributing_factor_vehicle_5': Utf8,
     'collision_id': Int64,
     'vehicle_type_code1': Utf8,
     'vehicle_type_code2': Utf8,
     'vehicle_type_code_3': Utf8,
     'vehicle_type_code_4': Utf8,
     'vehicle_type_code_5': Utf8}



We can see that the datatypes of Polars are built on top of [Arrow's datatypes](https://arrow.apache.org/docs/python/api/datatypes.html) and use Arrow arrays. This is awesome because Arrow is memory efficient and can also used for in-memory dataframes with zero-serialization across languages.

The first command I tried with Polars was looking for duplicates in the dataframe. I found I could do this with the syntax:


```python
test = (df.groupby("collision_id")
           .count()
           .filter(pl.col("count") > 1))

test
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (0, 2)</small><table border="1" class="dataframe"><thead><tr><th>collision_id</th><th>count</th></tr><tr><td>i64</td><td>u32</td></tr></thead><tbody></tbody></table></div>



Right away from the syntax I was in love.

Then I saw statements returned a dataframe:


```python
type(test)
```




    polars.dataframe.frame.DataFrame



This is exactly what I want! I don't want a series (even though Polars does have [Series](https://pola-rs.github.io/polars/py-polars/html/reference/series/index.html) data structures). You can even print the dataframes:


```python
print(test)
```

    shape: (0, 2)
    ┌──────────────┬───────┐
    │ collision_id ┆ count │
    │ ---          ┆ ---   │
    │ i64          ┆ u32   │
    ╞══════════════╪═══════╡
    └──────────────┴───────┘


This turns out to be helpful when you have lazy execution (which I'll go over later). The next thing I tried was to access the column of the dataframe by using the dot operator:


```python
df.crash_date
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[8], line 1
    ----> 1 df.crash_date


    AttributeError: 'DataFrame' object has no attribute 'crash_date'


I was actually happy to see this was not implemented! For me a column in a dataframe should not be accessed this way. The dot operator is meant to access attributes of the class.

Instead we can access the column of the dataframe like a dictionary's key:


```python
df["crash_date"].is_null().any()
```




    False



The crash dates are strings that I wanted to convert to datetime type (I'm doing this to build up to more complex queries). I can see the format of the string:


```python
df['crash_date'][0] # the .loc method doesnt exist!
```




    '2021-09-11T00:00:00.000'



To do so, I write two queries:

1. The first query extracts the year-month-day and writes it as a string in the format YYYY-MM-DD
2. The second query converts the YYYY-MM-DD strings into timestamp objects

For the first query I can extract the year-month-day from the string and assign that to a new column named `crash_date_str`. Note the syntax to create a new column in Polars is [with_columns](https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.with_columns.html) (similar to [withColumn](https://sparkbyexamples.com/pyspark/pyspark-withcolumn/) in Spark) and I have to use the [col](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.col.html) function similar to Spark! I can get the first 10 characters of the string using the vectorized [str method](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.html) similar to Pandas. Finally, I rename the new column `crash_data_str` using the [alias](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.alias.html) function (again just like Spark). The default for the `with_column` is to label the new column name the same as the old column name, so we use alias to rename it. 

In the second query I use the vectorized string method [strptime](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.str.strptime.html) to convert the `crash_date_str` column to a PyArrow datetime object and rename that column `crash_date` (overriding the old column with this name). 

These two queries are chained together and the results are shown below.


```python
df = df.with_columns(
            pl.col("crash_date").str.slice(0, length=10).alias("crash_date_str")
      ).with_columns(
            pl.col("crash_date_str").str.strptime(
                pl.Datetime, "%Y-%m-%d", strict=False).alias("crash_date")
)

df.select(["crash_date", "crash_date_str"]).head()
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (5, 2)</small><table border="1" class="dataframe"><thead><tr><th>crash_date</th><th>crash_date_str</th></tr><tr><td>datetime[μs]</td><td>str</td></tr></thead><tbody><tr><td>2021-09-11 00:00:00</td><td>&quot;2021-09-11&quot;</td></tr><tr><td>2022-03-26 00:00:00</td><td>&quot;2022-03-26&quot;</td></tr><tr><td>2022-06-29 00:00:00</td><td>&quot;2022-06-29&quot;</td></tr><tr><td>2021-09-11 00:00:00</td><td>&quot;2021-09-11&quot;</td></tr><tr><td>2021-12-14 00:00:00</td><td>&quot;2021-12-14&quot;</td></tr></tbody></table></div>



Notice the [col](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.col.html) function in Polars lets me access derived columns that are not in the original dataframe. In Pandas to do the same operations I would have to use a lambda function within an assign function:

    df.assign(crash_date=lambda: df["crash_date_str"].str.strptime(...))

I can see the number of crashes in each borough of NYC with the query


```python
print(df.groupby("borough").count())
```

    shape: (6, 2)
    ┌───────────────┬───────┐
    │ borough       ┆ count │
    │ ---           ┆ ---   │
    │ str           ┆ u32   │
    ╞═══════════════╪═══════╡
    │ MANHATTAN     ┆ 98    │
    │ STATEN ISLAND ┆ 27    │
    │ BROOKLYN      ┆ 247   │
    │ BRONX         ┆ 107   │
    │ null          ┆ 367   │
    │ QUEENS        ┆ 154   │
    └───────────────┴───────┘


There is a borough value of NULL. I can filter this out with the commands:


```python
nn_df = df.filter(pl.col("borough").is_not_null())
```

Now I can get just the unique values of non-null boroughs with the query: 


```python
print(df.filter(pl.col("borough").is_not_null())
        .select("borough")
        .unique())
```

    shape: (5, 1)
    ┌───────────────┐
    │ borough       │
    │ ---           │
    │ str           │
    ╞═══════════════╡
    │ STATEN ISLAND │
    │ MANHATTAN     │
    │ QUEENS        │
    │ BRONX         │
    │ BROOKLYN      │
    └───────────────┘


Notice that I can use the [select](https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.select.html) method in Polars to select just the columns I need. This is actually pretty powerful, as I can select columns and run queries on them similar to [selectEpr](https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.sql.DataFrame.selectExpr.html) in Spark:


```python
print(
 df.filter(pl.col("borough").is_not_null())
   .select([
       "borough", 
       (pl.col("number_of_persons_injured")  + 1).alias("number_of_persons_injured_plus1")
    ]).head()
)
```

    shape: (5, 2)
    ┌───────────┬─────────────────────────────────┐
    │ borough   ┆ number_of_persons_injured_plus1 │
    │ ---       ┆ ---                             │
    │ str       ┆ i64                             │
    ╞═══════════╪═════════════════════════════════╡
    │ BROOKLYN  ┆ 1                               │
    │ BROOKLYN  ┆ 1                               │
    │ BRONX     ┆ 3                               │
    │ BROOKLYN  ┆ 1                               │
    │ MANHATTAN ┆ 1                               │
    └───────────┴─────────────────────────────────┘


Doing the same query in Pandas is not as elegant or readable:


```python
(pd_df[~pd_df["borough"].isnull()]
      .assign(number_of_persons_injured_plus1=pd_df["number_of_persons_injured"] + 1)
      [["borough", "number_of_persons_injured_plus1"]]
      .head()
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>borough</th>
      <th>number_of_persons_injured_plus1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>BROOKLYN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BROOKLYN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BRONX</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BROOKLYN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>MANHATTAN</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



To me, the Polars query is so much easier to read. And what's more is that it's actually more efficient. The Pandas dataframe transforms the whole dataset, then subsets the columns to return just two. On the other hand Polars subsets the two columns first and then transforms just those two columns.

Now I can create a Polars dataframe the exact same way as in Pandas:


```python
borough_df = pl.DataFrame({
                "borough": ["BROOKLYN", "BRONX", "MANHATTAN", "STATEN ISLAND", "QUEENS"],
                "population": [2590516, 1379946, 1596273, 2278029, 378977],
                "area":[179.7, 109.2, 58.68, 281.6, 149.0]
})

print(borough_df)
```

    shape: (5, 3)
    ┌───────────────┬────────────┬───────┐
    │ borough       ┆ population ┆ area  │
    │ ---           ┆ ---        ┆ ---   │
    │ str           ┆ i64        ┆ f64   │
    ╞═══════════════╪════════════╪═══════╡
    │ BROOKLYN      ┆ 2590516    ┆ 179.7 │
    │ BRONX         ┆ 1379946    ┆ 109.2 │
    │ MANHATTAN     ┆ 1596273    ┆ 58.68 │
    │ STATEN ISLAND ┆ 2278029    ┆ 281.6 │
    │ QUEENS        ┆ 378977     ┆ 149.0 │
    └───────────────┴────────────┴───────┘


This is the population and area of the boroughs which I got from Wikipedia. I'll save it to s3. It was a little awkward to write to s3 with Polars directly so I'll first convert the dataframe to Pandas and then write to s3:





```python
borough_df.to_pandas().to_parquet("s3://harmonskis/nyc_populations.parquet")
```

However, reading from s3 is just the same as with Pandas:


```python
borough_df = pl.read_parquet("s3://harmonskis/nyc_populations.parquet")
```

We'll use it to go over a more complicated query:

    Get the total number of injuries per borough then join that result to the borough dataframe to get the injuries by population and finally sort them by borough name.

In Polars this can be using method chaining on the dataframe:


```python
print(
 df.filter(pl.col("borough").is_not_null())
   .select(["borough", "number_of_persons_injured"])
   .groupby("borough")
   .sum()
   .join(borough_df, on=["borough"])
   .select([
       "borough", 
       (pl.col("number_of_persons_injured") / pl.col("population")).alias("injuries_per_population")
   ])
   .sort(pl.col("borough"))
)
```

    shape: (5, 2)
    ┌───────────────┬─────────────────────────┐
    │ borough       ┆ injuries_per_population │
    │ ---           ┆ ---                     │
    │ str           ┆ f64                     │
    ╞═══════════════╪═════════════════════════╡
    │ BRONX         ┆ 0.000033                │
    │ BROOKLYN      ┆ 0.000045                │
    │ MANHATTAN     ┆ 0.000025                │
    │ QUEENS        ┆ 0.000193                │
    │ STATEN ISLAND ┆ 0.000007                │
    └───────────────┴─────────────────────────┘


Doing the same query in the Pandas API would be an awkward mess. As we can see in Polars it's very easy to use method chaining and the resulting syntax reads pretty similar to SQL! 

Which brings me to something that was super exciting to see in Polars: [sqlcontext](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.SQLContext.execute.html). SQLContext in Polars can be used to create a table from a Polars dataframe and then run SQL commands that return another Polars dataframe. 

We can see this by creating a table called `crashes` from the dataframe `df`:


```python
ctx = pl.SQLContext(crashes=df)
```

Now I can get the sum of every crash per day in each borough:


```python
daily_df = ctx.execute("""
    SELECT
        borough,
        crash_date AS day,
        SUM(number_of_persons_injured)
    FROM 
        crashes
    WHERE 
        borough IS NOT NULL
    GROUP BY 
        borough, crash_date
    ORDER BY 
        borough, day
""")

print(daily_df.collect().head())
```

    shape: (5, 3)
    ┌─────────┬─────────────────────┬───────────────────────────┐
    │ borough ┆ day                 ┆ number_of_persons_injured │
    │ ---     ┆ ---                 ┆ ---                       │
    │ str     ┆ datetime[μs]        ┆ i64                       │
    ╞═════════╪═════════════════════╪═══════════════════════════╡
    │ BRONX   ┆ 2021-02-26 00:00:00 ┆ 0                         │
    │ BRONX   ┆ 2021-04-06 00:00:00 ┆ 0                         │
    │ BRONX   ┆ 2021-04-08 00:00:00 ┆ 0                         │
    │ BRONX   ┆ 2021-04-10 00:00:00 ┆ 4                         │
    │ BRONX   ┆ 2021-04-11 00:00:00 ┆ 0                         │
    └─────────┴─────────────────────┴───────────────────────────┘


Notice I had to use `collect()` function to get the results. That is because by default SQL in Polars uses lazy execution.

You can see evidence of this when printing the resulting dataframe; it actually prints the query plan:


```python
print(daily_df)
```

    naive plan: (run LazyFrame.explain(optimized=True) to see the optimized plan)
    
    SORT BY [col("borough"), col("day")]
       SELECT [col("borough"), col("crash_date").alias("day"), col("number_of_persons_injured")] FROM
        AGGREGATE
        	[col("number_of_persons_injured").sum()] BY [col("borough"), col("crash_date")] FROM
          FILTER col("borough").is_not_null() FROM
          DF ["crash_date", "crash_time", "borough", "zip_code"]; PROJECT */30 COLUMNS; SELECTION: "None"


To get back a Polars dataframe from this result I would have to use the `eager=True` parameter in the execute method.

I can register this new dataframe as a table called `daily_crashes` in the SQLContext:


```python
ctx = ctx.register("daily_crashes", daily_df)
```

I can see the tables that are registered using the command:


```python
ctx.tables()
```




    ['crashes', 'daily_crashes']



Now say I want to get the current day's number of injured people and the prior days; I could use the [lag](https://www.sqlshack.com/sql-lag-function-overview-and-examples/) function in SQL to do so:


```python
ctx.execute("""
    SELECT
        borough,
        day,
        number_of_persons_injured,
        LAG(1,number_of_persons_injured) 
            OVER (
            PARTITION BY borough 
            ORDER BY day ASC
            ) AS prior_day_injured
FROM
    daily_crashes
ORDER BY 
    borough,
    day DESC
""", eager=True)
```


    ---------------------------------------------------------------------------

    InvalidOperationError                     Traceback (most recent call last)

    Cell In[26], line 1
    ----> 1 ctx.execute("""
          2     SELECT
          3         borough,
          4         day,
          5         number_of_persons_injured,
          6         LAG(1,number_of_persons_injured) 
          7             OVER (
          8             PARTITION BY borough 
          9             ORDER BY day ASC
         10             ) AS prior_day_injured
         11 FROM
         12     daily_crashes
         13 ORDER BY 
         14     borough,
         15     day DESC
         16 """, eager=True)


    File /opt/conda/lib/python3.10/site-packages/polars/sql/context.py:282, in SQLContext.execute(self, query, eager)
        204 def execute(self, query: str, eager: bool | None = None) -> LazyFrame | DataFrame:
        205     """
        206     Parse the given SQL query and execute it against the registered frame data.
        207 
       (...)
        280     └────────┴─────────────┴─────────┘
        281     """
    --> 282     res = wrap_ldf(self._ctxt.execute(query))
        283     return res.collect() if (eager or self._eager_execution) else res


    InvalidOperationError: unsupported SQL function: lag


I finally hit snag in Polars: their doesnt seem to be a lot of support for Window functions. This was initially disappointing since the library was so promising!

Upon further research I found window functions are supported, infact they are [**VERY WELL supported!**](https://pola-rs.github.io/polars-book/user-guide/expressions/window/). The query I was trying to turns out to be fairly easy to write as dataframe operations using the [over](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.over.html) expression. This is exactly the same as SQL where the column names within the `over(...)` operator are the columns you partition by. You can the sort within each partition (or group as they say in Polars) and use shift instead of LAG:


```python
print(
    daily_df.with_columns(
            pl.col("number_of_persons_injured")
              .sort_by("day", descending=False)
              .shift(periods=1)
              .over("borough")
              .alias("prior_day_injured")
).collect().head(8))
```

    shape: (8, 4)
    ┌─────────┬─────────────────────┬───────────────────────────┬───────────────────┐
    │ borough ┆ day                 ┆ number_of_persons_injured ┆ prior_day_injured │
    │ ---     ┆ ---                 ┆ ---                       ┆ ---               │
    │ str     ┆ datetime[μs]        ┆ i64                       ┆ i64               │
    ╞═════════╪═════════════════════╪═══════════════════════════╪═══════════════════╡
    │ BRONX   ┆ 2021-02-26 00:00:00 ┆ 0                         ┆ null              │
    │ BRONX   ┆ 2021-04-06 00:00:00 ┆ 0                         ┆ 0                 │
    │ BRONX   ┆ 2021-04-08 00:00:00 ┆ 0                         ┆ 0                 │
    │ BRONX   ┆ 2021-04-10 00:00:00 ┆ 4                         ┆ 0                 │
    │ BRONX   ┆ 2021-04-11 00:00:00 ┆ 0                         ┆ 4                 │
    │ BRONX   ┆ 2021-04-12 00:00:00 ┆ 0                         ┆ 0                 │
    │ BRONX   ┆ 2021-04-13 00:00:00 ┆ 3                         ┆ 0                 │
    │ BRONX   ┆ 2021-04-14 00:00:00 ┆ 3                         ┆ 3                 │
    └─────────┴─────────────────────┴───────────────────────────┴───────────────────┘


It turns out you can do the same thing with Pandas as shown below.

Note that I have to collect the lazy datafame and convert it to Pandas first:


```python
pd_daily_df = daily_df.collect().to_pandas()
```


```python
pd_daily_df = pd_daily_df.assign(prior_day_injured=
                        pd_daily_df.sort_values(by=['day'], ascending=True)
                          .groupby(['borough'])
                          ['number_of_persons_injured']
                          .shift(1))

pd_daily_df.head(8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>borough</th>
      <th>day</th>
      <th>number_of_persons_injured</th>
      <th>prior_day_injured</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BRONX</td>
      <td>2021-02-26</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BRONX</td>
      <td>2021-04-06</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BRONX</td>
      <td>2021-04-08</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BRONX</td>
      <td>2021-04-10</td>
      <td>4</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BRONX</td>
      <td>2021-04-11</td>
      <td>0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BRONX</td>
      <td>2021-04-12</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BRONX</td>
      <td>2021-04-13</td>
      <td>3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BRONX</td>
      <td>2021-04-14</td>
      <td>3</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



Syntactically, I still perfer the Polars to Pandas.

But let's I really want to use SQL and not do things in the dataframe, atleast to me, it doesnt seem possible with Polars.

Luckily there is another library that support blazingly fast SQL queries and integrates with Polars (and Pandas) directly: DuckDB.

## DuckDB To The Rescue For SQL <a class="anchor" id="fourth-bullet"></a>

I heard about [DuckDB](https://duckdb.org/) when I saw someone star it on github and thought it was "Yet Another SQL Engine". While DuckDB is a SQL engine, it does much more than I thought a SQL engine could! 

DuckDB is a parallel query processing library written in C++ and according to their website:

        DuckDB is designed to support analytical query workloads, also known as Online analytical processing (OLAP). These workloads are characterized by complex, relatively long-running queries that process significant portions of the stored dataset, for example aggregations over entire tables or joins between several large tables.
        ...
        DuckDB contains a columnar-vectorized query execution engine, where queries are still interpreted, but a large batch of values (a “vector”) are processed in one operation.

In other words, DuckDB can be used for fast SQL query execution on large datasets. For example the above query that failed in Polars runs perfectly using DuckDB:


```python
import duckdb

query = duckdb.sql("""
    SELECT
        borough,
        day,
        number_of_persons_injured,
        LAG(1, number_of_persons_injured) 
            OVER (
                PARTITION BY borough 
                ORDER BY day ASC
                ) as prior_day_injured
FROM
    daily_df
ORDER BY 
    borough,
    day DESC
LIMIT 5
""")
```

Now we can see the output of the query:


```python
query
```




    ┌─────────┬─────────────────────┬───────────────────────────┬───────────────────┐
    │ borough │         day         │ number_of_persons_injured │ prior_day_injured │
    │ varchar │      timestamp      │           int64           │       int32       │
    ├─────────┼─────────────────────┼───────────────────────────┼───────────────────┤
    │ BRONX   │ 2022-04-24 00:00:00 │                         0 │                 1 │
    │ BRONX   │ 2022-03-26 00:00:00 │                         7 │                 1 │
    │ BRONX   │ 2022-03-25 00:00:00 │                         1 │                 1 │
    │ BRONX   │ 2022-03-24 00:00:00 │                         1 │                 1 │
    │ BRONX   │ 2022-03-22 00:00:00 │                         1 │                 1 │
    └─────────┴─────────────────────┴───────────────────────────┴───────────────────┘



We can return the result as polars dataframe using the `pl` method:


```python
day_prior_df = query.pl()
print(day_prior_df.head(5))
```

    shape: (5, 4)
    ┌─────────┬─────────────────────┬───────────────────────────┬───────────────────┐
    │ borough ┆ day                 ┆ number_of_persons_injured ┆ prior_day_injured │
    │ ---     ┆ ---                 ┆ ---                       ┆ ---               │
    │ str     ┆ datetime[μs]        ┆ i64                       ┆ i32               │
    ╞═════════╪═════════════════════╪═══════════════════════════╪═══════════════════╡
    │ BRONX   ┆ 2022-04-24 00:00:00 ┆ 0                         ┆ 1                 │
    │ BRONX   ┆ 2022-03-26 00:00:00 ┆ 7                         ┆ 1                 │
    │ BRONX   ┆ 2022-03-25 00:00:00 ┆ 1                         ┆ 1                 │
    │ BRONX   ┆ 2022-03-24 00:00:00 ┆ 1                         ┆ 1                 │
    │ BRONX   ┆ 2022-03-22 00:00:00 ┆ 1                         ┆ 1                 │
    └─────────┴─────────────────────┴───────────────────────────┴───────────────────┘


Now we can see another cool part of DuckDB, you can execute SQL directly on local files!

First we save the daily crash dataframe as [Parquet](https://parquet.apache.org/)  file, but first remember it's a "lazy dataframe":


```python
daily_df
```




<i>naive plan: (run <b>LazyFrame.explain(optimized=True)</b> to see the optimized plan)</i>
    <p></p>
    <div>SORT BY [col("borough"), col("day")]<p></p>   SELECT [col("borough"), col("crash_date").alias("day"), col("number_of_persons_injured")] FROM<p></p>    AGGREGATE<p></p>    	[col("number_of_persons_injured").sum()] BY [col("borough"), col("crash_date")] FROM<p></p>      FILTER col("borough").is_not_null() FROM<p></p>      DF ["crash_date", "crash_time", "borough", "zip_code"]; PROJECT */30 COLUMNS; SELECTION: "None"</div>



It turns out you cant write lazy dataframes as Parquet using Polars. So first we'll collect it and then write it to parquet:


```python
daily_df.collect().write_parquet("daily_crashes.parquet")
```

[Apache Parquet](https://parquet.apache.org/) is a compressed columnar-stored file format that is great for analytical queries. Column-based formats are particularly good for [OLAP](https://aws.amazon.com/what-is/olap/) queries since columns can subsetted and be read in continuously allowing for aggregations to be easily performed on them. The datatypes for each column in Parquet are known which allows the format to be compressed. Since the columns and datatypes are known metadata we can read them in with the following query:


```python
duckdb.sql("SELECT * FROM parquet_schema(daily_crashes.parquet)").pl()
```





<div style="overflow-x: auto;">
<style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (4, 11)</small><table border="1" class="dataframe"><thead><tr><th>file_name</th><th>name</th><th>type</th><th>type_length</th><th>repetition_type</th><th>num_children</th><th>converted_type</th><th>scale</th><th>precision</th><th>field_id</th><th>logical_type</th></tr><tr><td>str</td><td>str</td><td>str</td><td>str</td><td>str</td><td>i64</td><td>str</td><td>i64</td><td>i64</td><td>i64</td><td>str</td></tr></thead><tbody><tr><td>&quot;daily_crashes.…</td><td>&quot;root&quot;</td><td>null</td><td>null</td><td>null</td><td>3</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td></tr><tr><td>&quot;daily_crashes.…</td><td>&quot;borough&quot;</td><td>&quot;BYTE_ARRAY&quot;</td><td>null</td><td>&quot;OPTIONAL&quot;</td><td>null</td><td>&quot;UTF8&quot;</td><td>null</td><td>null</td><td>null</td><td>&quot;StringType()&quot;</td></tr><tr><td>&quot;daily_crashes.…</td><td>&quot;day&quot;</td><td>&quot;INT64&quot;</td><td>null</td><td>&quot;OPTIONAL&quot;</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>&quot;TimestampType(…</td></tr><tr><td>&quot;daily_crashes.…</td><td>&quot;number_of_pers…</td><td>&quot;INT64&quot;</td><td>null</td><td>&quot;OPTIONAL&quot;</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td></tr></tbody></table></div>



Now we can perform queries on the actualy files without having to resort to dataframes at all:


```python
query = duckdb.sql("""
    SELECT
        borough,
        day,
        number_of_persons_injured,
        SUM(number_of_persons_injured) 
            OVER (
                PARTITION BY borough 
                ORDER BY day ASC
                ) AS cumulative_injuried
    FROM 
        read_parquet(daily_crashes.parquet)
    ORDER BY
        borough,
        day ASC
""")
```


```python
print(query.pl().head(8))
```

    shape: (8, 4)
    ┌─────────┬─────────────────────────┬───────────────────────────┬─────────────────────┐
    │ borough ┆ day                     ┆ number_of_persons_injured ┆ cumulative_injuried │
    │ ---     ┆ ---                     ┆ ---                       ┆ ---                 │
    │ str     ┆ str                     ┆ i64                       ┆ f64                 │
    ╞═════════╪═════════════════════════╪═══════════════════════════╪═════════════════════╡
    │ BRONX   ┆ 2021-02-26T00:00:00.000 ┆ 0                         ┆ 0.0                 │
    │ BRONX   ┆ 2021-04-06T00:00:00.000 ┆ 0                         ┆ 0.0                 │
    │ BRONX   ┆ 2021-04-08T00:00:00.000 ┆ 0                         ┆ 0.0                 │
    │ BRONX   ┆ 2021-04-10T00:00:00.000 ┆ 4                         ┆ 4.0                 │
    │ BRONX   ┆ 2021-04-11T00:00:00.000 ┆ 0                         ┆ 4.0                 │
    │ BRONX   ┆ 2021-04-12T00:00:00.000 ┆ 0                         ┆ 4.0                 │
    │ BRONX   ┆ 2021-04-13T00:00:00.000 ┆ 3                         ┆ 7.0                 │
    │ BRONX   ┆ 2021-04-14T00:00:00.000 ┆ 3                         ┆ 10.0                │
    └─────────┴─────────────────────────┴───────────────────────────┴─────────────────────┘


Pretty cool!!!

## Conclusions <a class="anchor" id="fifth-bullet"></a>

In this post I quickly covered what I view as the limitations of Pandas library. Next I covered how to get set up in with 
Jupyter lab using [Docker](https://www.docker.com/) on [AWS](https://aws.amazon.com/) and covered some basics of [Polars](https://www.pola.rs/), [DuckDB](https://duckdb.org/) and how to use the two in combination. The benefits of Polars is that,

* It allows for fast parallel querying on dataframes.
* It uses Apache Arrow for backend datatypes making it memory efficient.
* It has both lazy and eager execution mode.
* It allows for SQL queries directly on dataframes.
* Its API is similar to Spark's API and allows for highly readable queries using method chaining.

I am still new to both libraries, but looking forward to learning more about them.

Hope you enjoyed reading this!
