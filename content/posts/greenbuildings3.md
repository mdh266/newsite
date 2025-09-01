+++
authors = ["Mike Harmon"]
title = "Green Buildings 3: Build & Deploy Models With MLflow & Docker"
date = "2020-04-23"
tags = [
    "Scikit-Learn",
    "MLflow",
    "XGBoost",
    "Docker",
    "Google Cloud",
    "Regression"
]
series = ["Scikit-Learn"]
aliases = ["migrate-from-jekyl"]
+++

## Contents
-------

__[1. Introduction](#intro)__

__[2. Intro To MLflow](#mlflow-one)__

__[3. Linear Regression & Logging A Simple Run](#mlflow-two)__

__[4. XGBoost & Logging Nested Runs for GridSearchCV](#mlflow-three)__

__[5. MLflow Models: Model Serving With REST APIs](#mlflow-four)__

__[6. Deploying to Google App Engine with Docker](#mlflow-five)__

__[7. Conclusions ](#fifth-bullet)__


--------------

## Introduction <a class="anchor" id="intro"></a>
-------------


This is the third and final post in a series of blog posts about energy usage and green house gas emissions of buildings in New York City. In the [first post](https://mdh266.github.io/posts/greenbuildings1/) I covered exploratory data analysis and outlier removal.  In the [second post](https://mdh266.github.io/posts/greenbuildings2/) I covered imputing missing values. These topics make up the majority of what is called "data cleaning".  This last post will deal with model building and model deployment. Specifically I will build a model of New York City building green house gas emissions based on the building energy usage metrics. After I build a sufficiently accurate model I will convert the model to [REST API](https://restfulapi.net/) for serving and then deploy the REST API to the cloud.

The processes of model development and deployment are made a lot easier with [MLflow](https://mlflow.org/) library. Specifically, I will cover using the [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html) framework to log all the diffent models I developed as well as their performance.  MLflow tracking acts a great way to memorialize and document the model development process. I will then use [MLflow Models](https://www.mlflow.org/docs/latest/models.html) to convert the top model into a [REST API](https://restfulapi.net/) for model serving. I will go over two ways MLflow Models creates REST API including the newly added method that uses [Docker](https://www.docker.com/). Finally I will show how to simply deploy the "Dockerized" API to the cloud through [Google App Engine](https://cloud.google.com/appengine). 

Note the MLflow library is still *relatively new* and the APi may change, for this purpose I should remark that I am working with MLflow version 1.8.0.  I should also point out that model serving through Docker was [experimental](https://www.mlflow.org/docs/latest/cli.html#mlflow-models-build-docker) in MLflow 1.8.0 and may have changed since I finished this project.

## Working With MLflow <a class="anchor" id="mlflow-one"></a>
-----------------

[MLflow](https://mlflow.org/) is an open source tool to make machine learning easier and more reproducible and was create by [Databricks](https://databricks.com/) (the same people who created [Apache Spark](https://spark.apache.org/)). There are many components to MLflow, but the two I will be looking at are,

- [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html) : A tool for logging modeling experiments
- [MLflow Models](https://www.mlflow.org/docs/latest/models.html) : A tool for serving models as REST APIs

I will stick to using MLflow locally instead of a production set up. You can start the Web UI with the command:

    mlflow ui --host=0.0.0.0 --port=5050

Then going to the website http://0.0.0.0:5050 in your web broswer where we will see the following:


<img src="https://github.com/mdh266/NYCBuildingEnergyUse/blob/master/notebooks/images/EmptyMLFlow.png?raw=1">


We can see the generic MLflow website without any modeling experiment data. This will change soon enough. We can collect modeling information into "*experiments*" that will contain "*runs*". Each run could be one model or a series of different models each trained with different parameter values. In this way MLflow tracking is great for organizing and maintaining as much information about the development process as you like.  

Locally, MLflow will create a file directory called,

    mlruns
    
that will be housed in the same path that the `mlflow ui` was run in.  Let's import the library along with some other basic libraries:


```python
import mlflow
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()
```

We can see what the MLflow tracking url is to see where all the data for MLflow will be stored (i.e. where `mlruns` directory is):


```python
mlflow.get_tracking_uri()
```




    'file:///Users/mukeharmon/Documents/DS_Projects/NYCEnergyUsage/notebook/mlruns'



Now let's create an experiment for this project and get started!  


```python
try:
    mlflow.create_experiment("greenbuildings")
    experiment = mlflow.get_experiment_by_name("greenbuildings")
except:
    experiment = mlflow.get_experiment_by_name("greenbuildings")
```

We can see that we get an "experiment" that has a number of attributes:


```python
print(experiment)
```

    <Experiment: artifact_location='file:///Users/mukeharmon/Documents/DS_Projects/NYCEnergyUsage/notebook/mlruns/1', experiment_id='1', lifecycle_stage='active', name='greenbuildings', tags={}>


These attributes include:

- **artifact_location** (where the metadata + models will be stored)
- **experiment_id** (id to help us track the experiment)
- **lifestyle_stage** (whether its active, deleted, etc.)
- **name** (experiment name)
- **tag**


The `experiment_id` is an important attribute and will be used quite frequently to know where to log and organize all the modeling information. Let's set that number as a variable to use later:


```python
exp_id = experiment.experiment_id
```

Let's move on to building our first model for predicting green house gas emission of buildings.

## Linear Regression & Logging A Simple Run <a class="anchor" id="mlflow-two"></a>

Let's build a predictive model for green house gas emissions by multifamily homes and offices  in New York City.  We'll do this at first using a simple linear regression model. While not the best in terms of predictive performance it is often a great first step since it allows us to interpet the effect each feature has on the predicted green house gas emissions. We'll discuss this more later, but for now lets import our data from [Google BigQuery](https://cloud.google.com/bigquery) using the set up from the [previous posts](https://mdh266.github.io/posts/greenbuildings1):


```python
from google.oauth2 import service_account
from google.cloud import bigquery
import json
import pandas_gbq 


credentials = service_account.Credentials\
                             .from_service_account_file('./derby.json')

pandas_gbq.context.credentials = credentials
pandas_gbq.context.project     = credentials.project_id
```


```python
df = pandas_gbq.read_gbq("""
SELECT  
    CAST(Energy_Star AS INT64) AS Energy_Star,
    Site_EUI,
    NGI,
    EI,
    GHGI,
    CAST(Residential AS INT64) AS Residential,
FROM 
    db_gb.clean_data
""")
```

And get the target variable and features:


```python
X = df.drop("GHGI",axis=1)
Y = df["GHGI"] 
```

Let's remind ourselves what the distribution of the target variable and predictors look like using the pairplot shown in the [last post](https://mdh266.github.io/posts/greenbuildings2):


```python
sns.pairplot(df,
             vars=["Energy_Star","Site_EUI","NGI","EI","GHGI"],
             size=2,
             hue='Residential')
```




    <seaborn.axisgrid.PairGrid at 0x10f5ac160>




    
![png](/greenbuildings3_files/greenbuildings3_18_1.png)
    


**We can from the last row in this graph that the relationship between `GHGI` and `Site_EUI`, `NGI`, as well as `EI` is somewhat linear, but the relationship of `GHGI` and `Energy_Star` is less well defined.**

Let's create our train and test set as well fix our random state (to have repeatable datasets)


```python
from sklearn.model_selection import train_test_split

random_state = 93

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)
```

As we stated earlier we'll start out with a <a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">Linear Regression model</a> since it is simple and interpertable. We can easily implement a least squares regression model using Scikit-learn:


```python
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer

cat_feats = ['Residential']
num_feats = ['Energy_Star','Site_EUI','NGI','EI']

scaler = ColumnTransformer(transformers=[('num_transform', StandardScaler(), num_feats)],
                           remainder='passthrough')

pipe = Pipeline([('preprocessor', transformer),
                 ('reg', LinearRegression())])

model = pipe.fit(X_train, y_train)
```

Notice we use a [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) to only normalize the numerical features and not `Residential` which is categorical.  We can then evaluate the model performance ($R^{2}$-score) on the test set to *see how much variance in the model we are able to explain* as well as the mean square error (MSE):


```python
y_pred = model.predict(X_test)

print("R2 score: {}".format(r2_score(y_test, y_pred)))
print("MSE: {}".format(mean_squared_error(y_test, y_pred)))
```

    R2 score: 0.6476491751039197
    MSE: 2.226459421390599e-06


We can explain 64.76% of the variance which is pretty good, but definitely leaves room for improvement.  Let's take a look at the coefficients of the linear regression model.


```python
# get the last stage of the pipeline which is the model
reg = pipe.steps[1][1]

# print the coefficients
print(pd.Series(reg.coef_,index=X_train.columns))
```

    Energy_Star   -0.000124
    Site_EUI       0.001640
    NGI            0.000244
    EI             0.001017
    Residential    0.000268
    dtype: float64


The model coefficents can be interpreated as folllows: for continuous feautres, an increase in one of their in units yields an increase in the unit of green house emissions that is equal to the coefficent. For example, increasing `Site_EUI` by 1 unit increase `GHGI` by 0.00164 units. We can see that increasing the electricty, energy intensity, and natural gas intensity increases green house gas emissions which makes sense. Increasing the Energy Star rating of the building tends to decrease the greenhouse gas emissions which makes sense. It also seems that residential buildings tend to emit more green house gases than office space buildings, albiet weakly.

We can measure the p-values for coefficents by using Scikit-Learns's <a href="http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html">f_regression</a> function.


```python
from sklearn.feature_selection import f_regression

f_stats, pvals = f_regression(scaler.fit_transform(X_train), y_train)

print("P-Values:")
print(pd.Series(pvals,index=X_train.columns))
```

    P-Values:
    Energy_Star     0.000000e+00
    Site_EUI        0.000000e+00
    NGI            1.940368e-183
    EI              0.000000e+00
    Residential     6.788885e-28
    dtype: float64


**We see that even though the coeficients of the regression model are rather small, their small p-values show that they are still signifcant and should be included in our model.**  Overfitting a linear model can be quite obvious from the coefficients when one of the features has a large absolute value. In our model this does not seem to be the case and we don't have to consider overfitting or using regularization further.

Let's add a run to the MLflow experiment that corresponds to this model.  We use the `start_run` function and pass the experiment id along with the name for this run being "Linear Regression"


```python
run = mlflow.start_run(experiment_id=exp_id, run_name="Linear Regression")
```

We can see that we have an active run that is a [RunInfo](https://www.mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.RunInfo) entity that maintains information about the run:


```python
run.info
```




    <RunInfo: artifact_uri='file:///Users/mukeharmon/Documents/DS_Projects/NYCEnergyUsage/notebook/mlruns/1/835633c6abc4436ea913f471a123e729/artifacts', end_time=None, experiment_id='1', lifecycle_stage='active', run_id='835633c6abc4436ea913f471a123e729', run_uuid='835633c6abc4436ea913f471a123e729', start_time=1587943715502, status='RUNNING', user_id='mukeharmon'>



We can add the metrics for our model using the `add_metrics` functions:


```python
mlflow.log_metric("r2" ,r2_score(y_test, y_pred))
mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
```

Let's look at some of the residuals in the continuous features to see if we can find any non-linear patterns that might signal ways improve the model.


```python
from utilities.PlottingFunctions import plot_residuals

f = plot_residuals(X_test = X_test,
                   y_test = y_test,
                   y_pred = y_pred)
```


    
![png](/greenbuildings3_files/greenbuildings3_37_0.png)
    


There are no obvious patterns in the residuals, but at the same time they **do not appear to be normally distributed as the theory says they should be.**  This tells me that we might be able to use a more flexible model to capture the nonlinearities in the relationships.

We can log this image as well using the `log_artifact` method:


```python
f.savefig("resid.png")
mlflow.log_artifact("resid.png")
```

For the time being let's log the model using the so called [scikit-learn flavor](https://mlflow.org/docs/latest/models.html#scikit-learn-sklearn) and end the run:


```python
import mlflow.sklearn
mlflow.sklearn.log_model(model, "LinearModel")
mlflow.end_run()
```

We can go to the MLflow UI to see that the run has been added with its metrics:
<img src="https://github.com/mdh266/NYCBuildingEnergyUse/blob/master/notebooks/images/LinearModel1.png?raw=1">

Clicking on the run we can see the model performance metrics, logged model and artifacts:

<img src="https://github.com/mdh266/NYCBuildingEnergyUse/blob/master/notebooks/images/LinearModel2.png?raw=1">

The `LinearModel` folder under the artifacts tab contains the [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) (`conda.yml`), the pickled model (`model.pkl`) and associated metadata (`MLModel`). We should note that the `conda.yml` file is used to package all the necessary libraries for serving the model `model.pkl`.

##  XGBoost & Logging Nested Runs for GridSearchCV <a class="anchor" id="mlflow-three"></a>

Let's try another model to see if we cant improve the $R^2$ score and MSE. One algorithm that performs quite well is [XGBoost](https://xgboost.readthedocs.io/en/latest/). XGBoost is a based on gradient boosted decision trees and is one of the best performing machine learning models avaiable. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function. You can read more about gradient boosting [here](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/).


Let's import the XGBoost Regressor and then run a small grid search using cross-valiation to find the optimal parameter values:


```python
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# define the parameter values
paramters = {"n_estimators"   :[10,15,25,50,100,150],
             "max_depth"      :[3,5,7,10],
             "loss"           :["ls", "lad"]
}

# define the grid search and optimization metric
grid = GridSearchCV(estimator=XGBRegressor(),
                    param_grid=paramters,
                    scoring="r2",
                    cv=5,
                    n_jobs=-1)

# perform the grid search
xgb_model = grid.fit(X_train, y_train)
```

**NOTE: Scaling is unnecessary for tree based methods**.

Now that we have the best model from our grid search over the trainin set let see how it performs on the test set:


```python
y_pred = xgb_model.predict(X_test)

print("R^2 score: {}".format(r2_score(y_test, y_pred)))
print("MSE {}".format(mean_squared_error(y_test, y_pred)))
```

    R^2 score: 0.7761477343001326
    MSE 1.4144935977206672e-06


A definite improvement in the $R^2$ score and MSE! Let's take a look at the residuals:


```python
f = plot_residuals(X_test = X_test,
                   y_test = y_test,
                   y_pred = y_pred)

f.savefig("resid.png")
```


    
![png](/greenbuildings3_files/greenbuildings3_49_0.png)
    


Not terribly different than the linear regression model, but we see less outliers in the relationship between `NGI` and the residuals. 


While we have improved our prediction capabilities, one draw back to more complex models like XGBoost is that they are less interperable. Despite this draw back, XGBoost still allows to the find the relative importance of the features:


```python
model = grid.best_estimator_

for coef in zip(X_train.columns, model.feature_importances_):
    print(coef)
```

    ('Energy_Star', 0.03286382)
    ('Site_EUI', 0.4422697)
    ('NGI', 0.26643634)
    ('EI', 0.24186888)
    ('Residential', 0.016561199)


While we know the importance of the features we don't now how they effect the `GHGI`, i.e. if the `NGI` does the `GHGI` go up or down? By how much does it go up or down?  Given what we see just from the above these are not questions the model can answer in general terms. This is what we mean by model interperability!


Now let's log all the information from the grid search and each of the models performances using a nested run using a "with" statement:


```python
with mlflow.start_run(
            experiment_id=exp_id,
            run_name="XGBoostRegressor",
            nested=True
):
    # Get the grid cell results
    cv_results = grid.cv_results_

    # loop over each of the parameters and log them along
    # with the metric and rank of the model
    for params, metric, rank in zip(cv_results['params'],
                                    cv_results['mean_test_score'],
                                    cv_results["rank_test_score"]):
        
        with mlflow.start_run(experiment_id=exp_id, 
                              nested=True):
            # log the parameters
            mlflow.log_params(params)
            
            # log the R2 score
            mlflow.log_metric("r2",metric)
            
            # set the rank
            mlflow.set_tag("rank",rank)
            
    # For the best estimator (xbg_model) 
    # let's log the parameters for the best model and
    # its metric artifacts
    mlflow.log_params(grid.best_params_)
    mlflow.log_metrics({"r2" : r2_score(y_test, y_pred),
                       " mse": mean_squared_error(y_test, y_pred)})
    
    mlflow.log_artifact("resid.png")
    mlflow.log_model("XGBoost",xbg_model)
```

Let's take a look at the MLflow UI again:

<img src="https://github.com/mdh266/NYCBuildingEnergyUse/blob/master/notebooks/images/XGBoostRun.png?raw=1">

We can see the + symbol on the left side of the `Run Name` corresponding to the run "XGBoostRegresor".  Clicking on the symbol shows the rest of the results from the grid search as a dropdown:

![image.png](images/XGBoostGrid.png)

We can see the different parameters used in each of the runs, the $R^2$ value for that model as well as the ranking of that model. Notice that the first model and the last of the models with rank 1 are the same. However, their $R^2$ values are different. This is because the one is the model perfomance on the test set, while the other is the average model of the model performances in the 5-fold cross validation.

Another nice feature of MLflow is the ability to compare model runs, to see the effect of say a hyper-parameter on model performance. You can select the model runs from the drop down and click the "compare" button displayed in the pictures above. I took a sample from the grid search above to see the effect of `max_depth` on the model with 100 estimators and `ls` for its log function:


<img src="https://github.com/mdh266/NYCBuildingEnergyUse/blob/master/notebooks/images/Compare.png?raw=1">



We can see that the $R^{2}$ decreases as the `max_depth` increases. This makes sense as taking larger values of the `max_depth` generally leads to overfitting.

## MLflow Models: Serving With REST APIs & Docker<a class="anchor" id="mlflow-four"></a>

Now that we have built a good model for green house gas emission let's to deploy this model. One popular mechanism for deploying (or serving) a model is using a [REST API](https://restfulapi.net/). Deploying a model as an API means that we create a [webserver](https://en.wikipedia.org/wiki/Web_server) with a [url](https://en.wikipedia.org/wiki/URL) that accepts requests. End users or "clients" can make requests to the API and pass a list of data points containing features (usually as [json](https://www.json.org/json-en.html)). This list of features is fed into the model and the model spits out a list of predictions corresponding to those features. These list of predictions are sent back to the client (again usually as json).

The first step in the process is to save the model using the [XGBoost](https://www.mlflow.org/docs/latest/python_api/mlflow.xgboost.html) module:


```python
import mlflow.xgboost

model = mlflow.xgboost.load_model("/Users/mukeharmon/Documents/DS_Projects/NYCEnergyUsage/notebook/model/XGBModel")
```


```python
import mlflow
try:
    mlflow.create_experiment("some")
    experiment = mlflow.get_experiment_by_name("greenbuildings")
except:
    
```


```python
experiment = mlflow.create_experiment("some2")
```

    INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
    INFO  [alembic.runtime.migration] Will assume non-transactional DDL.



```python
experiment
```




    '2'




```python
mlflow.tracking.set_tracking_uri("sqlite:////Users/mukeharmon/Documents/DS_Projects/NYCEnergyUsage/notebook/mlflow.db")
```


```python
mlflow.get_artifact_uri()
```

    INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
    INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
    INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
    INFO  [alembic.runtime.migration] Will assume non-transactional DDL.





    './runs/0/bc4b93d03e4e49baaa31f00a4609223f/artifacts'




```python
run = mlflow.start_run(experiment_id=experiment)
mlflow.log_metric("mse",2)
mlflow.xgboost.log_model(model,"XGBoost")
mlflow.end_run()
```

    INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
    INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
    INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
    INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
    INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
    INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
    INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
    INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
    INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
    INFO  [alembic.runtime.migration] Will assume non-transactional DDL.



```python
mlflow.end_run()
```

    INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
    INFO  [alembic.runtime.migration] Will assume non-transactional DDL.


This creats a folder similar to the `LinerModel` one we discussed above. The next step is to use the MLflow commnand line to serve the model. From the directory where we saved the above model we can use the command:

    mlflow models serve --m XGBModel

This will initially show the following:
<img src="https://github.com/mdh266/NYCBuildingEnergyUse/blob/master/notebooks/images/XGBoostServe1.png?raw=1">

If everything builds properly we will then see the following:

<img src="https://github.com/mdh266/NYCBuildingEnergyUse/blob/master/notebooks/images/XGBoostServe1_P2.png?raw=1">

Notice the "Listening at: http://127.0.0.1:5000", this is the url for our webserver. We will make requests to get model predictions at the url. We have built a REST API that uses [flask](https://flask.palletsprojects.com/en/1.1.x/) and [gunicorn](https://gunicorn.org/) with a one line command using MLflow! To see how difficult this be to do by hand see my other [github repo](https://github.com/mdh266/DockerMLRestAPI).

Let's get some test data to try out model REST API:


```python
test_df = X_test.head(2)
test_df
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
      <th>Energy_Star</th>
      <th>Site_EUI</th>
      <th>NGI</th>
      <th>EI</th>
      <th>Residential</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3208</th>
      <td>7744</td>
      <td>52.7</td>
      <td>45.964180</td>
      <td>10.120456</td>
      <td>1</td>
    </tr>
    <tr>
      <th>103</th>
      <td>9</td>
      <td>112.2</td>
      <td>25.918749</td>
      <td>29.770710</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Let's get the predictions that the XGBoost model gives us as is to what the REST API returns:


```python
xgb_model.predict(test_df)
```




    array([0.00314277, 0.00720516], dtype=float32)



Now let's convert the test data from a [Pandas](https://pandas.pydata.org/) dataframe to json:


```python
test_json = test_df.to_json(orient='split')
test_json
```




    '{"columns":["Energy_Star","Site_EUI","NGI","EI","Residential"],"index":[3208,103],"data":[[7744,52.7,45.9641802469,10.1204555556,1],[9,112.2,25.9187489356,29.7707095517,1]]}'



We can submit the json as request for predictions to the REST API:


```python
import requests

result = requests.post(url="http://127.0.0.1:5000/invocations",
                       data=test_json,
                       headers={'Content-Type':'application/json'})
```


```python
result.json()
```




    [0.0031427741050720215, 0.007205158472061157]



The results are the same! Now let's take this one step further and use MLflow to build a [Docker](https://www.docker.com/) image so that we can deploy our REST API as a container. This again, is only one lineand that command is,
    
    mlflow models build-docker -m XGBModel -n xgbmodel
    
Where the `xgbmodel` is the tag for the Docker image and `XGBModel` is the folder we saved our model as. If the image is built properly we can see the following:

<img src="https://github.com/mdh266/NYCBuildingEnergyUse/blob/master/notebooks/images/Success.png?raw=1" >

The image is `fafed3745d54` and the tag is `xgbmodel:latest`. We can start up our containerized REST API using the command:

    docker run -ip 8000:8080 fafed3745d54
    
The `-p 8000:8080` is for [port forwarding](https://runnable.com/docker/binding-docker-ports). Notice that we have to use 8000 because the the results show:

<img src="https://github.com/mdh266/NYCBuildingEnergyUse/blob/master/notebooks/images/DockerRun.png?raw=1">

We can the make a request to that url and port:


```python
result = requests.post(url="http://127.0.0.1:8000/invocations",
                       data=test_json,
                       headers={'Content-Type':'application/json'})

result.json()
```




    [0.0031427741050720215, 0.007205158472061157]



The reults are the same as expected!

There is one downside for using MLflow to build a docker image and that is your image turns out to be quite large. We can see this from the command:

    docker images 
    
which shows,

<img src="https://github.com/mdh266/NYCBuildingEnergyUse/blob/master/notebooks/images/DockerImageSize.png?raw=1" >

Our model takes up 2.72 GB!  

## 6. Deploying to Google App Engine with Docker  <a class="anchor" id="mlflow-five"></a>

We have come to the last topic of this post which is deploying our model as REST API to the cloud. We can easily deploy the "Dockerized" model API to [Google Cloud App](https://cloud.google.com/appengine) Engine using the Docker image we created.   

The first step is to follow the instructions [here](https://cloud.google.com/container-registry/docs/quickstart) for copying the local Docker image to [Google Cloud Registry (GCR)](https://cloud.google.com/container-registry). For me the command was :

<img src="https://github.com/mdh266/NYCBuildingEnergyUse/blob/master/notebooks/images/pushtogcr.png?raw=1">

Once that is done we can check GCR to make sure the image has been pushed, you can see the results below:

<img src="https://github.com/mdh266/NYCBuildingEnergyUse/blob/master/notebooks/images/GCR.png?raw=1">

Next I built the `app.yaml` for a custom runtime and using flexibly environment as described [here](https://cloud.google.com/appengine/docs/flexible/custom-runtimes/build).  The contents of my `app.yaml` are:

    runtime: custom
    env: flex
    service: xgbmodel
    env_variables: 
      DISABLE_NGINX: "true"
      
It's important to note using the container will start [nginx](https://www.nginx.com/) and [gunicorn](https://gunicorn.org/) processes which we DO NOT want and therefore chose `DISABLE_NGINX: "true"` as discussed [here](https://www.mlflow.org/docs/latest/cli.html#mlflow-models-build-docker).


I then ran the command to deploy the app (`gcloud app deploy`) using `--image-url` with the address for my image in GCR:

![image](images/deploy.png)


One everything is completed I can check my app was created in the App Engine tab:

![image.png](images/AppEngine.png)

Now I can use the `target url` as pictured above to run a request against as shown below:


```python
target_url = "https://xgbmodel-dot-advance-sonar-232016.uc.r.appspot.com/invocations"

result = requests.post(url     = target_url,
                       data    = test_json,
                       headers = {'Content-Type':'application/json'})

result.json()
```




    [0.0031427741050720215, 0.007205158472061157]



It worked! To see how difficult this be to do by hand see my other [github repo](https://github.com/mdh266/DockerMLRestAPI).  That's enough for this post!

--------------
## Conclusions <a class="anchor" id="fifth-bullet"></a>
--------------

This blog post ends the series of blog posts that starts with a real life dataset on building energy usage and green house gas emissions. In previous posts we covered cleaning the dataset, but in this blog post we covered using the cleaned dataset and building a model with $R^2 \, = \, 0.776$ and deploying it to Google App Engine as an API using Docker. This process was made signifcantly easier using the MLflow library for model development and serving. This project was a lot of fun and I learned a ton workng on it.  I hope you found this useful!
