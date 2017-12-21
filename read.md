## High level design explanation
The server code is basically a Flask application with a single endpoint that servers a machine learning model. 
The app has a single blueprint for the API (/api), and inside a namespace for the Underwriting project (/api/underwriting), and finally a single endpoint inside this namespace (/api/underwriting/predict) which accepts a POST request with a JSON containing the fields needed for the model prediction. Once it gets a request, it extracts the fields from the JSON, creates a Pandas dataframe with them, and feed it to the model to obtain the prediction, which is then returned.

The model is loaded when the Flask app is first launched using Scikit learn library.

## 1 use object oriented coding style

   It is very easy to blew out your python code in a big project. A well-designed code, should be scalable and easy to extend. Object oriented programming is just for this purpose. Suppose we have three AI models, random forest, XGboost and Neural network. A recommended way to code is to design an abstract basic class.

   ```
    from abc importABCMeta,abstractmethod

    class Model(metaclass=ABCMeta):

    def __init__(self, start_date_training, end_date_training, feature_cols):

        pass

    @abstractmethod
    def predict(self):
       pass   

    @abstractmethod
    def train(self):
          pass  


  class mymodel(Model):

  def train():

      do_something()

  def predict():

      do_something()     

   ```
You can directly subclass the abstract base class. If you register your concerete class to ABC, you won't be able to call methods in base class from your subclass. For example, if you use model.register(mymodel), non-abstract class won't be visible in your virtual subclass, whichi is mymodel in this case.

## 2 Pandas DataFrame

### 2.1  use embed pandas function like apply to boost performance

  Pandas data frame has some optimized method. `Apply` is one typical example. Before you write big For loops, please think twice. Is there any recommended way to do this in Pandas?

 ```
    import numpy as np
    import pandas as pd
    dic = {'A':['1A','1A','3C','3C','3C','7M','7M','7M'],
       'B':[10,15,49,75,35,33,45,65],
       'C':[11,56,32,78,45,89,15,14],
       'D':[111,0,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
       'E':[0,222,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]}

   df = pd.DataFrame(dic)
  ```

```
    %%time
    df["transformed_1"] = df["C"].apply(lambda x:x+10)
```
    CPU times: user 859 µs, sys: 57 µs, total: 916 µs
    Wall time: 880 µs

```
    %%time
    for index,row in df.iterrows():
    row["transformed_2"] = row["C"]+10
````
    CPU times: user 9.33 ms, sys: 812 µs, total: 10.1 ms
    Wall time: 12 ms

##### apply function is `13` times faster than a for loop

```
    df.head()

    A	B	C	D	E	transformed	transformed_1
    0	1A	10	11	111.0	0.0	21	21
    1	1A	15	56	0.0	222.0	66	66
    2	3C	49	32	NaN	NaN	42	42
    3	3C	75	78	NaN	NaN	88	88
    4	3C	35	45	NaN	NaN	55	55
```

One thing to remind is to instead of using multiple apply function, you can use one apply function to return a Pandas `Series` object. Therefore, you are working on multiple Pandas columns in one scan of the whole data frame.

```
def procecss(df):

    square = df[x]^2
    cube   = df[x]^3

    return pd.Series({"cube":cube,"square":square})

df[["cube","square"]]=df.apply(lambda x:process(x),axis=1)
```   

### 2.2 joining data frames

You can work on Pandas data frames in SQL-like way to join and merge data frames.

```
pd.merge(left, right, how='inner', on=None, left_on="left", right_on="right")
```

https://pandas.pydata.org/pandas-docs/stable/merging.html

## 3 Database

run queries on database (DB)  and return aggregations

Try to run logic on DB and do aggregations there. Modern DB has been optimized in indexing and scaning data. It is smart to run SQL query on DB and return aggregations to client program. You can save both DB and network I/O that way.

If you want to find common information between two tables, consider `JOIN` instead of `Where`.


https://code.tutsplus.com/tutorials/writing-blazing-fast-mysql-queries--cms-25085

### 3.1 connect to MariaDB via mysqlclient

You may need to install the Python and MySQL development headers and libraries like so:

```brew install mysql-connector-c # macOS (Homebrew)```

then you can install from PyPI

```pip install mysqlclient```

developer API can be found at

https://mysqlclient.readthedocs.io/

## 4 Recommended Python Integreted Development Environment

Easy to do Debug and Profile your code
https://www.jetbrains.com/pycharm/

## 5 AWS
### 5.1 turn off your AWS service after your development

AWS keeps billing as soon as service is on.

### 6 GitHub

#### 6.1 work in your work branch and merge back to master branch

## 7 Data Processing

### 7.1 factorize your data

```
import pandas as pd

df = pd.DataFrame({'A':['type1','type2','type2'],
                   'B':['type1','type2','type3'],
                   'C':['type1','type3','type3']})
```
```
df
    A	    B	   C
0	type1	type1	type1
1	type2	type2	type3
2	type2	type3	type3
```

```
df = df.apply(lambda x: pd.factorize(x)[0])
```

```
  A B C
0	0	0	0
1	1	1	1
2	1	2	1
```                   
If you want numeric values for the same string value.

```
df = df.stack().rank(method='dense').unstack()
```
```
    A	 B	  C
0	1.0	1.0	1.0
1	2.0	2.0	3.0
2	2.0	3.0	3.0
```

### 7.2 find out all columns with the same prefix

A nice solution is to use regular expression. Also you can use a very straight forward way as the following.
```
med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
````

## 8 Install environment and launch API app

To install the needed libraries, you can go to the project root folder and run `conda env create`, this will
automatically create a new environment and install all the dependencies needed.

Once this is done, you can run `python api/application.py` to launch the app. You may need to add the project root folder to your
PYTHONPATH and also set the APPLICATION_MODE environment variable to either 'config.DevelopmentConfig', or
'config.ProductionConfig'
