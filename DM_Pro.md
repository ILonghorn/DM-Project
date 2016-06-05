
# Search Relevency Scorer

###### Import All the Basic Packages


```python
import pandas as pd 
import numpy as np
from __future__ import division
import re
import nltk
```

###### Read the all the Training Datasets


```python
train = pd.read_csv('train.csv')
attributes = pd.read_csv('attributes.csv')
prod_desc = pd.read_csv('product_descriptions.csv')
```

###### Merge the Product Title with Product Description to form combined Dataset


```python
new_train= pd.merge(prod_desc,train,how ='inner',on = 'product_uid')
new_train= new_train.drop('id', axis=1)
```

###### Save a Copy of the New Dataset


```python
pd.DataFrame.to_csv(new_train,'new_train.csv',index=False)
```

###### New Dataset


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
new_train.shape
```




    (74009, 21)



###### Correct the User Entered Search term using Dictionary Google Spell Check Api


```python
def spellcheck(self):
    if self in spell_check_dict:
        return spell_check_dict[self]
    else: 
        return self

new_train['search_correct'] = new_train.search_term.apply(spellcheck)
new_train = new_train.drop(['search_term'],axis=1)
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
      <th>relevance</th>
      <th>search_correct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>Not only do angles make joints stronger, they ...</td>
      <td>Simpson Strong-Tie 12-Gauge Angle</td>
      <td>3.00</td>
      <td>angle bracket</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>Not only do angles make joints stronger, they ...</td>
      <td>Simpson Strong-Tie 12-Gauge Angle</td>
      <td>2.50</td>
      <td>l bracket</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>BEHR Premium Textured DECKOVER is an innovativ...</td>
      <td>BEHR Premium Textured DeckOver 1-gal. #SC-141 ...</td>
      <td>3.00</td>
      <td>deck over</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>Update your bathroom with the Delta Vero Singl...</td>
      <td>Delta Vero 1-Handle Shower Only Faucet Trim Ki...</td>
      <td>2.33</td>
      <td>rain shower head</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>Update your bathroom with the Delta Vero Singl...</td>
      <td>Delta Vero 1-Handle Shower Only Faucet Trim Ki...</td>
      <td>2.67</td>
      <td>shower only faucet</td>
    </tr>
  </tbody>
</table>
</div>



### Refining the Search Term, Product description and Product Title

###### Tokenize the words and convert into lower case


```python
# let us define a function to carry out this process
def token(self):
    try:
        sen = self.lower()
        words = nltk.word_tokenize(sen)
        #sen = nltk.sent_tokenize(self)
        #for s in range(0,len(sen)):
            #words.append(nltk.word_tokenize(sen[s]))
    except:
        words = 0
    return words

new_train['search_words'] = new_train.search_correct.apply(token)
new_train['title_words'] = new_train.product_title.apply(token)
new_train['desc_words'] = new_train.product_description.apply(token)
```

###### Refined Training Data after Tokenizing


```python
new_train = new_train[new_train['title_words'] != 0]
# drop unwanted columns
new_train = new_train.drop(['product_description','product_title','search_correct'],axis=1)
new_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_uid</th>
      <th>relevance</th>
      <th>search_words</th>
      <th>title_words</th>
      <th>desc_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>3.00</td>
      <td>[angle, bracket]</td>
      <td>[simpson, strong-tie, 12-gauge, angle]</td>
      <td>[not, only, do, angles, make, joints, stronger...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>2.50</td>
      <td>[l, bracket]</td>
      <td>[simpson, strong-tie, 12-gauge, angle]</td>
      <td>[not, only, do, angles, make, joints, stronger...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>3.00</td>
      <td>[deck, over]</td>
      <td>[behr, premium, textured, deckover, 1-gal, ., ...</td>
      <td>[behr, premium, textured, deckover, is, an, in...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>2.33</td>
      <td>[rain, shower, head]</td>
      <td>[delta, vero, 1-handle, shower, only, faucet, ...</td>
      <td>[update, your, bathroom, with, the, delta, ver...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>2.67</td>
      <td>[shower, only, faucet]</td>
      <td>[delta, vero, 1-handle, shower, only, faucet, ...</td>
      <td>[update, your, bathroom, with, the, delta, ver...</td>
    </tr>
  </tbody>
</table>
</div>



###### Remove Irrelevant words that are Non Informative


```python
# importing list of stopwords from nltk
from nltk.corpus import stopwords

stop = stopwords.words('english')
def filteration(s):
    filtered = []
    for w in s:
        if w not in stop:
            filtered.append(w)
    return filtered

new_train['title_filter'] = new_train.title_words.apply(filteration)
new_train['desc_filter'] = new_train.desc_words.apply(filteration)
```

    /home/ishan/anaconda2/lib/python2.7/site-packages/ipykernel/__main__.py:8: UnicodeWarning: Unicode equal comparison failed to convert both arguments to Unicode - interpreting them as being unequal
    

###### New Training Data after Filtering out STOPWORDS


```python
# drop unwanted columns
new_train = new_train.drop(['title_words','desc_words'],axis=1)
new_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_uid</th>
      <th>relevance</th>
      <th>search_words</th>
      <th>title_filter</th>
      <th>desc_filter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>3.00</td>
      <td>[angle, bracket]</td>
      <td>[simpson, strong-tie, 12-gauge, angle]</td>
      <td>[angles, make, joints, stronger, ,, also, prov...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>2.50</td>
      <td>[l, bracket]</td>
      <td>[simpson, strong-tie, 12-gauge, angle]</td>
      <td>[angles, make, joints, stronger, ,, also, prov...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>3.00</td>
      <td>[deck, over]</td>
      <td>[behr, premium, textured, deckover, 1-gal, ., ...</td>
      <td>[behr, premium, textured, deckover, innovative...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>2.33</td>
      <td>[rain, shower, head]</td>
      <td>[delta, vero, 1-handle, shower, faucet, trim, ...</td>
      <td>[update, bathroom, delta, vero, single-handle,...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>2.67</td>
      <td>[shower, only, faucet]</td>
      <td>[delta, vero, 1-handle, shower, faucet, trim, ...</td>
      <td>[update, bathroom, delta, vero, single-handle,...</td>
    </tr>
  </tbody>
</table>
</div>



###### We have observed that lot of words contain '-' and are not easy to deal with. Hence, splitting words at '-'


```python
# define a function to split words at the hyphen
def splitt(s):
    words = []
    for w in s:
        if '-' in w:
            lst = re.split('-',w)
            for l in lst:
                words.append(l)
        else:
            words.append(w)
    return words

new_train['title_filter2'] = new_train.title_filter.apply(splitt)
new_train['desc_filter2'] = new_train.desc_filter.apply(splitt)
new_train['search_filter2'] = new_train.search_words.apply(splitt)
```

###### Training Data after the Split at '-'


```python
# dropping old filters
new_train = new_train.drop(['title_filter','desc_filter','search_words'],axis=1)
new_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_uid</th>
      <th>relevance</th>
      <th>title_filter2</th>
      <th>desc_filter2</th>
      <th>search_filter2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>3.00</td>
      <td>[simpson, strong, tie, 12, gauge, angle]</td>
      <td>[angles, make, joints, stronger, ,, also, prov...</td>
      <td>[angle, bracket]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>2.50</td>
      <td>[simpson, strong, tie, 12, gauge, angle]</td>
      <td>[angles, make, joints, stronger, ,, also, prov...</td>
      <td>[l, bracket]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>3.00</td>
      <td>[behr, premium, textured, deckover, 1, gal, .,...</td>
      <td>[behr, premium, textured, deckover, innovative...</td>
      <td>[deck, over]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>2.33</td>
      <td>[delta, vero, 1, handle, shower, faucet, trim,...</td>
      <td>[update, bathroom, delta, vero, single, handle...</td>
      <td>[rain, shower, head]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>2.67</td>
      <td>[delta, vero, 1, handle, shower, faucet, trim,...</td>
      <td>[update, bathroom, delta, vero, single, handle...</td>
      <td>[shower, only, faucet]</td>
    </tr>
  </tbody>
</table>
</div>



###### Standardization of Numbers in dataset to numerical form


```python
#Code for standardization of numbers
def text2int(self, numwords={}):
    if not numwords:
      units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]

      tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

      scales = ["hundred", "thousand", "million", "billion", "trillion"]

      numwords["and"] = (1, 0)
      for idx, word in enumerate(units):    numwords[word] = (1, idx)
      for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
      for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    flag = 0
    count = 0
    a = []
    for word in self:
        if word not in numwords:
            if flag == 0:
                a.append(word)
            else:
                a.append(result+current)
                a.append(word)
                flag = 0
        if word in numwords:
            if word == 'and' and flag == 0:
                a.append(word)
            else:
                flag = 1
                count = count + 1
                scale, increment = numwords[word]
                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
    if flag == 1:
        a.append(result+current)

    return a
```


```python
new_train['std_search'] = new_train.search_filter2.apply(text2int)
new_train['std_title'] = new_train.title_filter2.apply(text2int)
new_train['std_desc'] = new_train.desc_filter2.apply(text2int)
```


```python
new_train = new_train.drop(['search_filter2', 'title_filter2','desc_filter2'],axis = 1)
new_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_uid</th>
      <th>relevance</th>
      <th>std_search</th>
      <th>std_title</th>
      <th>std_desc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>3.00</td>
      <td>[angle, bracket]</td>
      <td>[simpson, strong, tie, 12, gauge, angle]</td>
      <td>[angles, make, joints, stronger, ,, also, prov...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>2.50</td>
      <td>[l, bracket]</td>
      <td>[simpson, strong, tie, 12, gauge, angle]</td>
      <td>[angles, make, joints, stronger, ,, also, prov...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>3.00</td>
      <td>[deck, over]</td>
      <td>[behr, premium, textured, deckover, 1, gal, .,...</td>
      <td>[behr, premium, textured, deckover, innovative...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>2.33</td>
      <td>[rain, shower, head]</td>
      <td>[delta, vero, 1, handle, shower, faucet, trim,...</td>
      <td>[update, bathroom, delta, vero, single, handle...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>2.67</td>
      <td>[shower, only, faucet]</td>
      <td>[delta, vero, 1, handle, shower, faucet, trim,...</td>
      <td>[update, bathroom, delta, vero, single, handle...</td>
    </tr>
  </tbody>
</table>
</div>



###### Doing Lemmatization to convert Plural nouns into Singular


```python
lemm = nltk.WordNetLemmatizer()
character = [',',",",'.','?','/','{','}','[',']','|',"!","@","$","%","&","*","(",")" ]
def lem(col):
    words = []
    for w in col:
        if w not in character:
             try:
                 word = lemm.lemmatize(w)
             except:
                 word = w
             words.append(word)
    return words

new_train['desc_lemm'] = new_train.std_desc.apply(lem)
new_train['title_lemm'] = new_train.std_title.apply(lem)
new_train['search_lemm'] = new_train.std_search.apply(lem)
```

###### Training Data After Lemmatization


```python
new_train = new_train.drop(['std_desc','std_title','std_search'],axis=1)
new_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_uid</th>
      <th>relevance</th>
      <th>desc_lemm</th>
      <th>title_lemm</th>
      <th>search_lemm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>3.00</td>
      <td>[angle, make, joint, stronger, also, provide, ...</td>
      <td>[simpson, strong, tie, 12, gauge, angle]</td>
      <td>[angle, bracket]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>2.50</td>
      <td>[angle, make, joint, stronger, also, provide, ...</td>
      <td>[simpson, strong, tie, 12, gauge, angle]</td>
      <td>[l, bracket]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>3.00</td>
      <td>[behr, premium, textured, deckover, innovative...</td>
      <td>[behr, premium, textured, deckover, 1, gal, #,...</td>
      <td>[deck, over]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>2.33</td>
      <td>[update, bathroom, delta, vero, single, handle...</td>
      <td>[delta, vero, 1, handle, shower, faucet, trim,...</td>
      <td>[rain, shower, head]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>2.67</td>
      <td>[update, bathroom, delta, vero, single, handle...</td>
      <td>[delta, vero, 1, handle, shower, faucet, trim,...</td>
      <td>[shower, only, faucet]</td>
    </tr>
  </tbody>
</table>
</div>



###### Finding Item Color in the Training Data


```python
# define function to identify the color
def color_present(self):    
    lst = []
    color_dict=['red','orange','yellow','green','blue','indigo','violet',
            'purple','magenta','cyan','pink','brown','white','gray',
            'black','chrome','golden','silver']
    for j in self:
        if j in color_dict and j not in lst:
            lst.append(j)
        color = lst
        if len(lst) == 0:
            color = 0
    return color
new_train['color_search']= new_train.search_lemm.apply(color_present)
new_train['color_desc']= new_train.desc_lemm.apply(color_present)
new_train['color_title'] = new_train.title_lemm.apply(color_present)
new_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_uid</th>
      <th>relevance</th>
      <th>desc_lemm</th>
      <th>title_lemm</th>
      <th>search_lemm</th>
      <th>color_search</th>
      <th>color_desc</th>
      <th>color_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>3.00</td>
      <td>[angle, make, joint, stronger, also, provide, ...</td>
      <td>[simpson, strong, tie, 12, gauge, angle]</td>
      <td>[angle, bracket]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>2.50</td>
      <td>[angle, make, joint, stronger, also, provide, ...</td>
      <td>[simpson, strong, tie, 12, gauge, angle]</td>
      <td>[l, bracket]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>3.00</td>
      <td>[behr, premium, textured, deckover, innovative...</td>
      <td>[behr, premium, textured, deckover, 1, gal, #,...</td>
      <td>[deck, over]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>2.33</td>
      <td>[update, bathroom, delta, vero, single, handle...</td>
      <td>[delta, vero, 1, handle, shower, faucet, trim,...</td>
      <td>[rain, shower, head]</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>2.67</td>
      <td>[update, bathroom, delta, vero, single, handle...</td>
      <td>[delta, vero, 1, handle, shower, faucet, trim,...</td>
      <td>[shower, only, faucet]</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
    </tr>
  </tbody>
</table>
</div>



###### Finding Item Material in Training Data


```python
# Define function to identify the Item Material
def mat_present(self):    
    lst=[]
    material_dict=['wood','steel','bronze','plastic','stainless','zinc','silicon','rubber','fibre','wax','powder','glass',
                  'copper']
    for j in self:
        if j in material_dict and j not in lst:
            lst.append(j)
        material = lst
        if len(lst) == 0:
            material = 0
    return material   
new_train['mat_search']= new_train.search_lemm.apply(mat_present)
new_train['mat_desc']= new_train.desc_lemm.apply(mat_present)
new_train['mat_title'] = new_train.title_lemm.apply(mat_present)
new_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_uid</th>
      <th>relevance</th>
      <th>desc_lemm</th>
      <th>title_lemm</th>
      <th>search_lemm</th>
      <th>color_search</th>
      <th>color_desc</th>
      <th>color_title</th>
      <th>mat_search</th>
      <th>mat_desc</th>
      <th>mat_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>3.00</td>
      <td>[angle, make, joint, stronger, also, provide, ...</td>
      <td>[simpson, strong, tie, 12, gauge, angle]</td>
      <td>[angle, bracket]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[zinc]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>2.50</td>
      <td>[angle, make, joint, stronger, also, provide, ...</td>
      <td>[simpson, strong, tie, 12, gauge, angle]</td>
      <td>[l, bracket]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[zinc]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>3.00</td>
      <td>[behr, premium, textured, deckover, innovative...</td>
      <td>[behr, premium, textured, deckover, 1, gal, #,...</td>
      <td>[deck, over]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[wood]</td>
      <td>[wood]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>2.33</td>
      <td>[update, bathroom, delta, vero, single, handle...</td>
      <td>[delta, vero, 1, handle, shower, faucet, trim,...</td>
      <td>[rain, shower, head]</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>2.67</td>
      <td>[update, bathroom, delta, vero, single, handle...</td>
      <td>[delta, vero, 1, handle, shower, faucet, trim,...</td>
      <td>[shower, only, faucet]</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



###### Carry out Stemming Operation to reduce words into root words to standardize all words 


```python
from nltk import PorterStemmer
ps = PorterStemmer()

# Doing the stemming and removing grams with leng￼th less than or equal to 2
def stemm(col):
    words = []
    for w in col:
        try:
            word = ps.stem(str(w))
        except:
            word = w
        words.append(word)
    return words

new_train['title_stem'] = new_train.title_lemm.apply(stemm)
new_train['desc_stem'] = new_train.desc_lemm.apply(stemm)
new_train['search_stem'] = new_train.search_lemm.apply(stemm)
```

    /home/ishan/anaconda2/lib/python2.7/site-packages/nltk/stem/porter.py:274: UnicodeWarning: Unicode equal comparison failed to convert both arguments to Unicode - interpreting them as being unequal
      if word[-1] == 's':
    

###### Training Data after STEMMING


```python
new_train = new_train.drop(['title_lemm','desc_lemm','search_lemm'],axis=1)
new_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_uid</th>
      <th>relevance</th>
      <th>color_search</th>
      <th>color_desc</th>
      <th>color_title</th>
      <th>mat_search</th>
      <th>mat_desc</th>
      <th>mat_title</th>
      <th>title_stem</th>
      <th>desc_stem</th>
      <th>search_stem</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[zinc]</td>
      <td>0</td>
      <td>[simpson, strong, tie, 12, gaug, angl]</td>
      <td>[angl, make, joint, stronger, also, provid, co...</td>
      <td>[angl, bracket]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>2.50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[zinc]</td>
      <td>0</td>
      <td>[simpson, strong, tie, 12, gaug, angl]</td>
      <td>[angl, make, joint, stronger, also, provid, co...</td>
      <td>[l, bracket]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[wood]</td>
      <td>[wood]</td>
      <td>[behr, premium, textur, deckov, 1, gal, #, sc,...</td>
      <td>[behr, premium, textur, deckov, innov, solid, ...</td>
      <td>[deck, over]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>2.33</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[delta, vero, 1, handl, shower, faucet, trim, ...</td>
      <td>[updat, bathroom, delta, vero, singl, handl, s...</td>
      <td>[rain, shower, head]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>2.67</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[delta, vero, 1, handl, shower, faucet, trim, ...</td>
      <td>[updat, bathroom, delta, vero, singl, handl, s...</td>
      <td>[shower, onli, faucet]</td>
    </tr>
  </tbody>
</table>
</div>



###### Now we have refined our Text data. Proceed to Attribute Creation

#### 1. Creating 'Brand' Name attribute


```python
def brand(self):
    try:
        name = self[0]
    except:
        name = self
    return name

new_train['Brand']= new_train.title_stem.apply(brand)
```

###### Training data with attribute 'Brand'


```python
new_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_uid</th>
      <th>relevance</th>
      <th>color_search</th>
      <th>color_desc</th>
      <th>color_title</th>
      <th>mat_search</th>
      <th>mat_desc</th>
      <th>mat_title</th>
      <th>title_stem</th>
      <th>desc_stem</th>
      <th>search_stem</th>
      <th>Brand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[zinc]</td>
      <td>0</td>
      <td>[simpson, strong, tie, 12, gaug, angl]</td>
      <td>[angl, make, joint, stronger, also, provid, co...</td>
      <td>[angl, bracket]</td>
      <td>simpson</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>2.50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[zinc]</td>
      <td>0</td>
      <td>[simpson, strong, tie, 12, gaug, angl]</td>
      <td>[angl, make, joint, stronger, also, provid, co...</td>
      <td>[l, bracket]</td>
      <td>simpson</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[wood]</td>
      <td>[wood]</td>
      <td>[behr, premium, textur, deckov, 1, gal, #, sc,...</td>
      <td>[behr, premium, textur, deckov, innov, solid, ...</td>
      <td>[deck, over]</td>
      <td>behr</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>2.33</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[delta, vero, 1, handl, shower, faucet, trim, ...</td>
      <td>[updat, bathroom, delta, vero, singl, handl, s...</td>
      <td>[rain, shower, head]</td>
      <td>delta</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>2.67</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[delta, vero, 1, handl, shower, faucet, trim, ...</td>
      <td>[updat, bathroom, delta, vero, singl, handl, s...</td>
      <td>[shower, onli, faucet]</td>
      <td>delta</td>
    </tr>
  </tbody>
</table>
</div>



###### 2. Create an attribute 'B.in.S' to identify whether customer have searched with 'Brand' or not 


```python
# define a function to carry out the task of looking brand in search words
def lookbrand(df):
    if df['Brand'] in df['search_stem']:
        return 1
    else:
        return 0

new_train['B.in.Search']= new_train.apply(lookbrand,axis=1)
```

###### New Training data with attribute 'B.in.S'


```python
new_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_uid</th>
      <th>relevance</th>
      <th>color_search</th>
      <th>color_desc</th>
      <th>color_title</th>
      <th>mat_search</th>
      <th>mat_desc</th>
      <th>mat_title</th>
      <th>title_stem</th>
      <th>desc_stem</th>
      <th>search_stem</th>
      <th>Brand</th>
      <th>B.in.Search</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[zinc]</td>
      <td>0</td>
      <td>[simpson, strong, tie, 12, gaug, angl]</td>
      <td>[angl, make, joint, stronger, also, provid, co...</td>
      <td>[angl, bracket]</td>
      <td>simpson</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>2.50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[zinc]</td>
      <td>0</td>
      <td>[simpson, strong, tie, 12, gaug, angl]</td>
      <td>[angl, make, joint, stronger, also, provid, co...</td>
      <td>[l, bracket]</td>
      <td>simpson</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[wood]</td>
      <td>[wood]</td>
      <td>[behr, premium, textur, deckov, 1, gal, #, sc,...</td>
      <td>[behr, premium, textur, deckov, innov, solid, ...</td>
      <td>[deck, over]</td>
      <td>behr</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>2.33</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[delta, vero, 1, handl, shower, faucet, trim, ...</td>
      <td>[updat, bathroom, delta, vero, singl, handl, s...</td>
      <td>[rain, shower, head]</td>
      <td>delta</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>2.67</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[delta, vero, 1, handl, shower, faucet, trim, ...</td>
      <td>[updat, bathroom, delta, vero, singl, handl, s...</td>
      <td>[shower, onli, faucet]</td>
      <td>delta</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



###### 3. Create an attribute for Search_term length. The number of words in the search_words will impact the quality of search result


```python
new_train['search_length']= new_train.search_stem.str.len()
new_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_uid</th>
      <th>relevance</th>
      <th>color_search</th>
      <th>color_desc</th>
      <th>color_title</th>
      <th>mat_search</th>
      <th>mat_desc</th>
      <th>mat_title</th>
      <th>title_stem</th>
      <th>desc_stem</th>
      <th>search_stem</th>
      <th>Brand</th>
      <th>B.in.Search</th>
      <th>search_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[zinc]</td>
      <td>0</td>
      <td>[simpson, strong, tie, 12, gaug, angl]</td>
      <td>[angl, make, joint, stronger, also, provid, co...</td>
      <td>[angl, bracket]</td>
      <td>simpson</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>2.50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[zinc]</td>
      <td>0</td>
      <td>[simpson, strong, tie, 12, gaug, angl]</td>
      <td>[angl, make, joint, stronger, also, provid, co...</td>
      <td>[l, bracket]</td>
      <td>simpson</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[wood]</td>
      <td>[wood]</td>
      <td>[behr, premium, textur, deckov, 1, gal, #, sc,...</td>
      <td>[behr, premium, textur, deckov, innov, solid, ...</td>
      <td>[deck, over]</td>
      <td>behr</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>2.33</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[delta, vero, 1, handl, shower, faucet, trim, ...</td>
      <td>[updat, bathroom, delta, vero, singl, handl, s...</td>
      <td>[rain, shower, head]</td>
      <td>delta</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>2.67</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[delta, vero, 1, handl, shower, faucet, trim, ...</td>
      <td>[updat, bathroom, delta, vero, singl, handl, s...</td>
      <td>[shower, onli, faucet]</td>
      <td>delta</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



###### 4. Create an attribute to calculate number of 'Search_words' in 'Product Title' and 'Product Description'


```python
# function to calculate prop of search words in title 
def word_prop(df):
    count = 0
    for w in df['search_stem']:
        if w in df['title_stem']:
            count = count+1
    div = count/len(df['search_stem'])        
    return div

# 'S_I_T'= proportion of search words in title
new_train['S_I_T']=new_train.apply(word_prop,axis=1)
```

    /home/ishan/anaconda2/lib/python2.7/site-packages/ipykernel/__main__.py:5: UnicodeWarning: Unicode equal comparison failed to convert both arguments to Unicode - interpreting them as being unequal
    


```python
# function to calculate prop of search words in desc 
def word_prop2(df):
    count = 0
    for w in df['search_stem']:
        if w in df['desc_stem']:
            count = count+1
    div = count/len(df['search_stem'])        
    return div

# 'S_I_D'= proportion of search words in desc
new_train['S_I_D']=new_train.apply(word_prop2,axis=1)
```

###### New Data set after attribute Creation


```python
new_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_uid</th>
      <th>relevance</th>
      <th>color_search</th>
      <th>color_desc</th>
      <th>color_title</th>
      <th>mat_search</th>
      <th>mat_desc</th>
      <th>mat_title</th>
      <th>title_stem</th>
      <th>desc_stem</th>
      <th>search_stem</th>
      <th>Brand</th>
      <th>B.in.Search</th>
      <th>search_length</th>
      <th>S_I_T</th>
      <th>S_I_D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[zinc]</td>
      <td>0</td>
      <td>[simpson, strong, tie, 12, gaug, angl]</td>
      <td>[angl, make, joint, stronger, also, provid, co...</td>
      <td>[angl, bracket]</td>
      <td>simpson</td>
      <td>0</td>
      <td>2</td>
      <td>0.500000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>2.50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[zinc]</td>
      <td>0</td>
      <td>[simpson, strong, tie, 12, gaug, angl]</td>
      <td>[angl, make, joint, stronger, also, provid, co...</td>
      <td>[l, bracket]</td>
      <td>simpson</td>
      <td>0</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[wood]</td>
      <td>[wood]</td>
      <td>[behr, premium, textur, deckov, 1, gal, #, sc,...</td>
      <td>[behr, premium, textur, deckov, innov, solid, ...</td>
      <td>[deck, over]</td>
      <td>behr</td>
      <td>0</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>2.33</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[delta, vero, 1, handl, shower, faucet, trim, ...</td>
      <td>[updat, bathroom, delta, vero, singl, handl, s...</td>
      <td>[rain, shower, head]</td>
      <td>delta</td>
      <td>0</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>2.67</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[delta, vero, 1, handl, shower, faucet, trim, ...</td>
      <td>[updat, bathroom, delta, vero, singl, handl, s...</td>
      <td>[shower, onli, faucet]</td>
      <td>delta</td>
      <td>0</td>
      <td>3</td>
      <td>0.666667</td>
      <td>0.666667</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_train['SUM'] = new_train['S_I_D'] + new_train['S_I_T']
```


```python
import matplotlib.pyplot as plt
```


```python
%matplotlib inline
plt.scatter(new_train['B.in.Search'],new_train['relevance'])
```




    <matplotlib.collections.PathCollection at 0x7f2ba94ec190>




![png](output_61_1.png)


###### Classify Relevance into 'Low(0)' and 'High(1)'


```python
# defining func for classification
def classify(self):
    if self > 2:
        return 1
    else: 
        return 0
new_train['Class'] = new_train.relevance.apply(classify)
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
      <th>relevance</th>
      <th>color_search</th>
      <th>color_desc</th>
      <th>color_title</th>
      <th>mat_search</th>
      <th>mat_desc</th>
      <th>mat_title</th>
      <th>title_stem</th>
      <th>desc_stem</th>
      <th>search_stem</th>
      <th>Brand</th>
      <th>B.in.Search</th>
      <th>search_length</th>
      <th>S_I_T</th>
      <th>S_I_D</th>
      <th>SUM</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[zinc]</td>
      <td>0</td>
      <td>[simpson, strong, tie, 12, gaug, angl]</td>
      <td>[angl, make, joint, stronger, also, provid, co...</td>
      <td>[angl, bracket]</td>
      <td>simpson</td>
      <td>0</td>
      <td>2</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>2.50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[zinc]</td>
      <td>0</td>
      <td>[simpson, strong, tie, 12, gaug, angl]</td>
      <td>[angl, make, joint, stronger, also, provid, co...</td>
      <td>[l, bracket]</td>
      <td>simpson</td>
      <td>0</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[wood]</td>
      <td>[wood]</td>
      <td>[behr, premium, textur, deckov, 1, gal, #, sc,...</td>
      <td>[behr, premium, textur, deckov, innov, solid, ...</td>
      <td>[deck, over]</td>
      <td>behr</td>
      <td>0</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>2.33</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[delta, vero, 1, handl, shower, faucet, trim, ...</td>
      <td>[updat, bathroom, delta, vero, singl, handl, s...</td>
      <td>[rain, shower, head]</td>
      <td>delta</td>
      <td>0</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.666667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>2.67</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[delta, vero, 1, handl, shower, faucet, trim, ...</td>
      <td>[updat, bathroom, delta, vero, singl, handl, s...</td>
      <td>[shower, onli, faucet]</td>
      <td>delta</td>
      <td>0</td>
      <td>3</td>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>1.333333</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.scatter(new_train['SUM'],new_train['Class'])
```




    <matplotlib.collections.PathCollection at 0x7f2ba91ec690>




![png](output_65_1.png)



```python
# define a function to create Color attribute
def lookcolor(df):
    if df['color_search'] == 0:
        return 1
    if df['color_search'] != 0:
        for item in df['color_search']:
            try:
                if item in ( df['color_desc'] or df['color_title']):
                    return 1
                    break
                else:
                    return 0
            except:
                return 0
new_train['color']= new_train.apply(lookcolor,axis=1)
```


```python
# define a function to create Material attribute
def lookmat(df):
    if df['mat_search'] == 0:
        return 1
    if df['mat_search'] != 0:
        for item in df['mat_search']:
            try:
                if item in ( df['mat_desc'] or df['mat_title']):
                    return 1
                    break
                else:
                    return 0
            except:
                return 0
new_train['material']= new_train.apply(lookmat,axis=1)
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
      <th>relevance</th>
      <th>color_search</th>
      <th>color_desc</th>
      <th>color_title</th>
      <th>mat_search</th>
      <th>mat_desc</th>
      <th>mat_title</th>
      <th>title_stem</th>
      <th>desc_stem</th>
      <th>search_stem</th>
      <th>Brand</th>
      <th>B.in.Search</th>
      <th>search_length</th>
      <th>S_I_T</th>
      <th>S_I_D</th>
      <th>SUM</th>
      <th>Class</th>
      <th>color</th>
      <th>material</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[zinc]</td>
      <td>0</td>
      <td>[simpson, strong, tie, 12, gaug, angl]</td>
      <td>[angl, make, joint, stronger, also, provid, co...</td>
      <td>[angl, bracket]</td>
      <td>simpson</td>
      <td>0</td>
      <td>2</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>2.50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[zinc]</td>
      <td>0</td>
      <td>[simpson, strong, tie, 12, gaug, angl]</td>
      <td>[angl, make, joint, stronger, also, provid, co...</td>
      <td>[l, bracket]</td>
      <td>simpson</td>
      <td>0</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[wood]</td>
      <td>[wood]</td>
      <td>[behr, premium, textur, deckov, 1, gal, #, sc,...</td>
      <td>[behr, premium, textur, deckov, innov, solid, ...</td>
      <td>[deck, over]</td>
      <td>behr</td>
      <td>0</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100005</td>
      <td>2.33</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[delta, vero, 1, handl, shower, faucet, trim, ...</td>
      <td>[updat, bathroom, delta, vero, singl, handl, s...</td>
      <td>[rain, shower, head]</td>
      <td>delta</td>
      <td>0</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.666667</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>2.67</td>
      <td>0</td>
      <td>[chrome]</td>
      <td>[chrome]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[delta, vero, 1, handl, shower, faucet, trim, ...</td>
      <td>[updat, bathroom, delta, vero, singl, handl, s...</td>
      <td>[shower, onli, faucet]</td>
      <td>delta</td>
      <td>0</td>
      <td>3</td>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>1.333333</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



###### Calculate Distance Matrix


```python
Bag =list(new_train.search_stem.sum())
```


```python
def make_unique(original_list):
    unique_list = []
    [unique_list.append(obj) for obj in original_list if obj not in unique_list]
    return unique_list
Bagg = make_unique(Bag)
```


```python
print len(Bagg)
print len(Bag)
```

    5874
    237302
    


```python
Bag_sorted = sorted(Bagg)
Bag_sort = Bag_sorted[2:]
```


```python
new_train2 = new_train[:10].copy()
```


```python
# define a function to filter Bagwords
def lookbag(df):
    vector1 = []
    vector2 = []
    vector3 = []
    for w in Bag_sort:
        if w in df['search_stem']:
            vector1.append(1)
        else:
            vector1.append(0)
    for w in Bag_sort:
        if w in df['title_stem']:
            vector2.append(1)
        else:
            vector2.append(0)
    for w in Bag_sort:
        if w in df['desc_stem']:
            vector3.append(1)
        else:
            vector3.append(0)
    vector = (np.asarray(vector2)+np.asarray(vector3))/2
    return np.linalg.norm(np.asarray(vector1)-vector)
    #df['search_desc'] = np.linalg.norm(np.asarray(vector1)-np.asarray(vector3)
new_train['Distance']= new_train.apply(lookbag,axis=1)
```

    /home/ishan/anaconda2/lib/python2.7/site-packages/ipykernel/__main__.py:12: UnicodeWarning: Unicode equal comparison failed to convert both arguments to Unicode - interpreting them as being unequal
    


```python
new_train3 = new_train.drop(['color_search','color_desc','color_title','mat_search','mat_desc','mat_title'
                             ,'relevance','product_uid','title_stem','desc_stem','search_stem','Brand'],axis=1)
```


```python
new_train3.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B.in.Search</th>
      <th>search_length</th>
      <th>S_I_T</th>
      <th>S_I_D</th>
      <th>SUM</th>
      <th>Class</th>
      <th>color</th>
      <th>material</th>
      <th>Distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4.213075</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4.444097</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4.743416</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.666667</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4.213075</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>1.333333</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3.968627</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame.to_csv(new_train3,'Final_data2.csv',index=False)
```


```python
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
```


```python
Fdata = pd.read_csv('Final_data.csv')
```


```python
Fdata.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B.in.Search</th>
      <th>search_length</th>
      <th>S_I_T</th>
      <th>S_I_D</th>
      <th>SUM</th>
      <th>Class</th>
      <th>color</th>
      <th>material</th>
      <th>Normal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.166338</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.452422</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.823079</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.666667</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.166338</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>1.333333</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-0.136370</td>
    </tr>
  </tbody>
</table>
</div>




```python
#new_train3['lnormal'] = new_train3.Distance.apply(np.log)
```


```python
Fdata = Fdata.drop(['SUM','Normal'],axis=1)
```


```python
Fdata.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B.in.Search</th>
      <th>search_length</th>
      <th>S_I_T</th>
      <th>S_I_D</th>
      <th>Class</th>
      <th>color</th>
      <th>material</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns
```


```python
#Fdata['B.in.Search'] = Fdata['B.in.Search'] + 1
Fdata['S_I_T'] = Fdata['S_I_T'] + 1
Fdata['S_I_D'] = Fdata['S_I_D'] + 1
Fdata['color'] = Fdata['material'] + 1
Fdata['material'] = Fdata['material'] + 1
#Fdata['Normal'] = Fdata['Normal'] + 1
```


```python
#Fdata['B.in.Search'] = Fdata.B.in.Search.apply(np.log)
Fdata['S_I_T'] = Fdata.S_I_T.apply(np.log)
Fdata['S_I_D'] = Fdata.S_I_D.apply(np.log)
Fdata['color'] = Fdata.color.apply(np.log)
Fdata['material'] = Fdata.material.apply(np.log)
#Fdata['Normal'] = Fdata.Normal.apply(np.log)
```


```python
Fdata.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B.in.Search</th>
      <th>search_length</th>
      <th>S_I_T</th>
      <th>S_I_D</th>
      <th>Class</th>
      <th>color</th>
      <th>material</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>0.405465</td>
      <td>0.405465</td>
      <td>1</td>
      <td>0.693147</td>
      <td>0.693147</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.693147</td>
      <td>0.693147</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.405465</td>
      <td>1</td>
      <td>0.693147</td>
      <td>0.693147</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>0.287682</td>
      <td>0.287682</td>
      <td>1</td>
      <td>0.693147</td>
      <td>0.693147</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0.510826</td>
      <td>0.510826</td>
      <td>1</td>
      <td>0.693147</td>
      <td>0.693147</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = Fdata.pop('Class')
X = Fdata.copy()
```


```python
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,stratify = y,random_state = 0)
```


```python
clf2 = RandomForestClassifier(n_estimators=500,n_jobs=-1,oob_score=True)
```


```python
Model = clf2.fit(X_train,y_train)
```


```python
print 'Training Accuracy:=', accuracy_score(y_train,Model.predict(X_train))*100
print
print 'Testing Accuracy:=', accuracy_score(y_test,Model.predict(X_test))*100
```

    Training Accuracy:= 70.1965023356
    
    Testing Accuracy:= 69.7653470252
    


```python
# Calculating accuracy using Majority class predictor
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='most_frequent')
```


```python
DUMB = dummy.fit(X_train,y_train)
print 'DUMB Training Accuracy:=', accuracy_score(y_train,DUMB.predict(X_train))*100
print
print 'DUMB Testing Accuracy:=', accuracy_score(y_test,DUMB.predict(X_test))*100
```

    DUMB Training Accuracy:= 68.0809172683
    
    DUMB Testing Accuracy:= 68.0808899698
    


```python
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
print accuracy_score(y_test,clf.predict(X_test))*100
print accuracy_score(y_train,clf.predict(X_train))*100
```

    69.6842768995
    70.1965023356
    
