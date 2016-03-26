

```python
import pandas as pd
```


```python
train = pd.read_csv('/home/ishan/PycharmProjects/Data Mining Project/train.csv')
```


```python
train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>product_uid</th>
      <th>product_title</th>
      <th>search_term</th>
      <th>relevance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>100001</td>
      <td>Simpson Strong-Tie 12-Gauge Angle</td>
      <td>angle bracket</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>100001</td>
      <td>Simpson Strong-Tie 12-Gauge Angle</td>
      <td>l bracket</td>
      <td>2.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>100002</td>
      <td>BEHR Premium Textured DeckOver 1-gal. #SC-141 ...</td>
      <td>deck over</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>100005</td>
      <td>Delta Vero 1-Handle Shower Only Faucet Trim Ki...</td>
      <td>rain shower head</td>
      <td>2.33</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17</td>
      <td>100005</td>
      <td>Delta Vero 1-Handle Shower Only Faucet Trim Ki...</td>
      <td>shower only faucet</td>
      <td>2.67</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 74067 entries, 0 to 74066
    Data columns (total 5 columns):
    id               74067 non-null int64
    product_uid      74067 non-null int64
    product_title    74067 non-null object
    search_term      74067 non-null object
    relevance        74067 non-null float64
    dtypes: float64(1), int64(2), object(2)
    memory usage: 3.4+ MB



```python
attributes = pd.read_csv('/home/ishan/PycharmProjects/Data Mining Project/attributes.csv')
```


```python
attributes.head(20)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_uid</th>
      <th>name</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>Bullet01</td>
      <td>Versatile connector for various 90° connection...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>Bullet02</td>
      <td>Stronger than angled nailing or screw fastenin...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100001</td>
      <td>Bullet03</td>
      <td>Help ensure joints are consistently straight a...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100001</td>
      <td>Bullet04</td>
      <td>Dimensions: 3 in. x 3 in. x 1-1/2 in.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100001</td>
      <td>Bullet05</td>
      <td>Made from 12-Gauge steel</td>
    </tr>
    <tr>
      <th>5</th>
      <td>100001</td>
      <td>Bullet06</td>
      <td>Galvanized for extra corrosion resistance</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100001</td>
      <td>Bullet07</td>
      <td>Install with 10d common nails or #9 x 1-1/2 in...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>100001</td>
      <td>Gauge</td>
      <td>12</td>
    </tr>
    <tr>
      <th>8</th>
      <td>100001</td>
      <td>Material</td>
      <td>Galvanized Steel</td>
    </tr>
    <tr>
      <th>9</th>
      <td>100001</td>
      <td>MFG Brand Name</td>
      <td>Simpson Strong-Tie</td>
    </tr>
    <tr>
      <th>10</th>
      <td>100001</td>
      <td>Number of Pieces</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>100001</td>
      <td>Product Depth (in.)</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>100001</td>
      <td>Product Height (in.)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>100001</td>
      <td>Product Weight (lb.)</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>14</th>
      <td>100001</td>
      <td>Product Width (in.)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>15</th>
      <td>100002</td>
      <td>Application Method</td>
      <td>Brush,Roller,Spray</td>
    </tr>
    <tr>
      <th>16</th>
      <td>100002</td>
      <td>Assembled Depth (in.)</td>
      <td>6.63 in</td>
    </tr>
    <tr>
      <th>17</th>
      <td>100002</td>
      <td>Assembled Height (in.)</td>
      <td>7.76 in</td>
    </tr>
    <tr>
      <th>18</th>
      <td>100002</td>
      <td>Assembled Width (in.)</td>
      <td>6.63 in</td>
    </tr>
    <tr>
      <th>19</th>
      <td>100002</td>
      <td>Bullet01</td>
      <td>Revives wood and composite decks, railings, po...</td>
    </tr>
  </tbody>
</table>
</div>




```python
attributes.info(verbose=True, null_counts=True)
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2044803 entries, 0 to 2044802
    Data columns (total 3 columns):
    product_uid    2044648 non-null float64
    name           2044648 non-null object
    value          2042713 non-null object
    dtypes: float64(1), object(2)
    memory usage: 62.4+ MB



```python
prod_desc = pd.read_csv('/home/ishan/PycharmProjects/Data Mining Project/product_descriptions.csv')
```


```python
prod_desc.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_uid</th>
      <th>product_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>Not only do angles make joints stronger, they ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002</td>
      <td>BEHR Premium Textured DECKOVER is an innovativ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100003</td>
      <td>Classic architecture meets contemporary design...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100004</td>
      <td>The Grape Solar 265-Watt Polycrystalline PV So...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>Update your bathroom with the Delta Vero Singl...</td>
    </tr>
  </tbody>
</table>
</div>




```python
prod_desc.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 124428 entries, 0 to 124427
    Data columns (total 2 columns):
    product_uid            124428 non-null int64
    product_description    124428 non-null object
    dtypes: int64(1), object(1)
    memory usage: 2.8+ MB



```python
# merge the datasets
```


```python
# first merging the product descriptions to the training data set
```


```python
new_train= pd.merge(prod_desc,train,how ='inner',on = 'product_uid')
```


```python
new_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_uid</th>
      <th>product_description</th>
      <th>id</th>
      <th>product_title</th>
      <th>search_term</th>
      <th>relevance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>Not only do angles make joints stronger, they ...</td>
      <td>2</td>
      <td>Simpson Strong-Tie 12-Gauge Angle</td>
      <td>angle bracket</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>Not only do angles make joints stronger, they ...</td>
      <td>3</td>
      <td>Simpson Strong-Tie 12-Gauge Angle</td>
      <td>l bracket</td>
      <td>2.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>BEHR Premium Textured DECKOVER is an innovativ...</td>
      <td>9</td>
      <td>BEHR Premium Textured DeckOver 1-gal. #SC-141 ...</td>
      <td>deck over</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>Update your bathroom with the Delta Vero Singl...</td>
      <td>16</td>
      <td>Delta Vero 1-Handle Shower Only Faucet Trim Ki...</td>
      <td>rain shower head</td>
      <td>2.33</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>Update your bathroom with the Delta Vero Singl...</td>
      <td>17</td>
      <td>Delta Vero 1-Handle Shower Only Faucet Trim Ki...</td>
      <td>shower only faucet</td>
      <td>2.67</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_train.shape
```




    (74067, 6)




```python
new_train= new_train.drop('id', axis=1)
```


```python
new_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_uid</th>
      <th>product_description</th>
      <th>product_title</th>
      <th>search_term</th>
      <th>relevance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>Not only do angles make joints stronger, they ...</td>
      <td>Simpson Strong-Tie 12-Gauge Angle</td>
      <td>angle bracket</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>Not only do angles make joints stronger, they ...</td>
      <td>Simpson Strong-Tie 12-Gauge Angle</td>
      <td>l bracket</td>
      <td>2.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>BEHR Premium Textured DECKOVER is an innovativ...</td>
      <td>BEHR Premium Textured DeckOver 1-gal. #SC-141 ...</td>
      <td>deck over</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>Update your bathroom with the Delta Vero Singl...</td>
      <td>Delta Vero 1-Handle Shower Only Faucet Trim Ki...</td>
      <td>rain shower head</td>
      <td>2.33</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>Update your bathroom with the Delta Vero Singl...</td>
      <td>Delta Vero 1-Handle Shower Only Faucet Trim Ki...</td>
      <td>shower only faucet</td>
      <td>2.67</td>
    </tr>
  </tbody>
</table>
</div>




```python
print new_train.loc[50,'product_description']
print 
print new_train.loc[50,'product_title']
print
print new_train.loc[50,'search_term']
print
print new_train.loc[50,'relevance']
```

    This easy-to-assemble Decorative Wire Chrome Finish Commercial Shelving Unit from HDX provides storage space for any room and purpose including commercial, industrial and residential use. The 6-tier shelving unit is made of tubular steel and wires coated with a durable chrome finish. All shelf heights are easily adjustable, and the unit can be assembled without any tools. Each shelf holds up to 600 lbs. when evenly distributed and features a total unit weight capacity of 3600 lbs. for optimal use in kitchens, pantries, utility rooms, warehouses and garages.Each shelf supports 600 lbs. when evenly distributed; total unit weight capacity is 3600 lbs.No tools required for easy assemblyTubular steel construction with durable chrome finish offers solid support for a variety of heavy-duty itemsPerfect storage solution for commercial or residential use in any room, office or jobsite6 fully adjustable shelves and legs help stabilize unit on uneven flooringNSF-certified for food storageNote: Product may vary by store.
    
    HDX 48 in. W x 72 in. H x 18 in. D Decorative Wire Chrome Finish Commercial Shelving Unit
    
    kitchen wire shelf tiered
    
    3.0



```python
pd.DataFrame.to_csv(new_train,'/home/ishan/PycharmProjects/Data Mining Project/new_train.csv')
```


```python
import nltk
```


```python
# Splitting the search terms into words
```


```python
from nltk.tokenize import word_tokenize
```


```python
# let us define a function to carry out this process
```


```python
def bagging(self):
    sen = self
    words = word_tokenize(sen)  
    return words
```


```python
new_train['search_words'] = new_train.search_term.apply(bagging)
#new_train['desc_words'] = new_train.product_description.apply(bagging)
#new_train['title_words'] = new_train.product_title.apply(bagging)
```


```python
new_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_uid</th>
      <th>product_description</th>
      <th>product_title</th>
      <th>search_term</th>
      <th>relevance</th>
      <th>search_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>Not only do angles make joints stronger, they ...</td>
      <td>Simpson Strong-Tie 12-Gauge Angle</td>
      <td>angle bracket</td>
      <td>3.00</td>
      <td>[angle, bracket]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>Not only do angles make joints stronger, they ...</td>
      <td>Simpson Strong-Tie 12-Gauge Angle</td>
      <td>l bracket</td>
      <td>2.50</td>
      <td>[l, bracket]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>BEHR Premium Textured DECKOVER is an innovativ...</td>
      <td>BEHR Premium Textured DeckOver 1-gal. #SC-141 ...</td>
      <td>deck over</td>
      <td>3.00</td>
      <td>[deck, over]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>Update your bathroom with the Delta Vero Singl...</td>
      <td>Delta Vero 1-Handle Shower Only Faucet Trim Ki...</td>
      <td>rain shower head</td>
      <td>2.33</td>
      <td>[rain, shower, head]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>Update your bathroom with the Delta Vero Singl...</td>
      <td>Delta Vero 1-Handle Shower Only Faucet Trim Ki...</td>
      <td>shower only faucet</td>
      <td>2.67</td>
      <td>[shower, only, faucet]</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
