# Stress-Predict-Project

Aplikasi `Stress Predict Project` merupakan web app sederhana yang dapat 
memprediksi tingkat stress seseorang melalui kelakuannya selama tertidur.

Web app dibangun berdasarkan algoritma *Decision Tree* dan *Support Vector Machines*
menggunakan dataset dari Smart Yoga Pillow (SaYoPillow) dari [kaggle](https://www.kaggle.com/datasets/laavanya/human-stress-detection-in-and-through-sleep).

Berikut penjelasan mengenai datasetnya.


## About Dataset (Human Stress Detection in and through Sleep) 

`Human Stress Detection in and through Sleep by monitoring physiological data.`

Considering today’s lifestyle, people just sleep forgetting the benefits sleep 
provides to the human body. Smart-Yoga Pillow (SaYoPillow) is proposed to help 
in understanding the relationship between stress and sleep and to fully materialize 
the idea of “Smart-Sleeping” by proposing an edge device. An edge processor with 
a model analyzing the physiological changes that occur during sleep along with 
the sleeping habits is proposed. Based on these changes during sleep, stress 
prediction for the following day is proposed. The secure transfer of the analyzed 
stress data along with the average physiological changes to the IoT cloud for 
storage is implemented. A secure transfer of any data from the cloud to any 
third-party applications is also proposed. A user interface is provided allowing 
the user to control the data accessibility and visibility. SaYoPillow is novel, 
with security features as well as consideration of sleeping habits for stress 
reduction, with an accuracy of up to 96%.

In SayoPillow.csv, you will see the relationship between the parameters- 
1. snoring range of the user 
2. respiration rate
3. body temperature
4. limb movement rate
5. blood oxygen levels
6. eye movement
7. number of hours of sleep
8. heart rate

    `predict value:`

9. Stress Levels:

    0-low/normal, 
    
    1-medium low, 
    
    2-medium, 
    
    3-medium high, 
    
    4-high

You can get the .csv file 
[here](https://www.kaggle.com/datasets/laavanya/human-stress-detection-in-and-through-sleep)

If you are using this dataset or found any of this information contributing 
towards your research, please cite: 
1.	L. Rachakonda, A. K. Bapatla, S. P. Mohanty, and E. Kougianos, “SaYoPillow: Blockchain-Integrated Privacy-Assured IoMT Framework for Stress Management Considering Sleeping Habits”, IEEE Transactions on Consumer Electronics (TCE), Vol. 67, No. 1, Feb 2021, pp. 20-29.
2.	L. Rachakonda, S. P. Mohanty, E. Kougianos, K. Karunakaran, and M. Ganapathiraju, “Smart-Pillow: An IoT based Device for Stress Detection Considering Sleeping Habits”, in Proceedings of the 4th IEEE International Symposium on Smart Electronic Systems (iSES), 2018, pp. 161--166. 

## Penjelasan Data

Proses pengolahan data untuk membuat model kali ini menggunakan bahasa 
pemrograman python melalui google colaboratory.

1. Import library yang akan digunakan.

~~~python
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from sklearn import metrics
~~~

2. Memasukkan data ke dalam variabel untuk diolah.
~~~python
data = "SaYoPillow.csv"
dataset = pd.read_csv(data)
~~~

3. Menampilkan tabel.
~~~python
dataset.head()
~~~

**output:**

~~~python
        sr	rr	t	lm	bo	rem	sr.1	hr	sl
0	93.80	25.680	91.840	16.600	89.840	99.60	1.840	74.20	3
1	91.64	25.104	91.552	15.880	89.552	98.88	1.552	72.76	3
2	60.00	20.000	96.000	10.000	95.000	85.00	7.000	60.00	1
3	85.76	23.536	90.768	13.920	88.768	96.92	0.768	68.84	3
4	48.12	17.248	97.872	6.496	96.248	72.48	8.248	53.12	0
~~~

4. Mengubah kolom `sr.1` menjadi sh (Sleep hours)
~~~python
dataset = dataset.rename(columns={'sr.1' : 'sh'})
dataset.head()
~~~

**output:**

~~~python
        sr	rr	t	lm	bo	rem	sh	hr	sl
0	93.80	25.680	91.840	16.600	89.840	99.60	1.840	74.20	3
1	91.64	25.104	91.552	15.880	89.552	98.88	1.552	72.76	3
2	60.00	20.000	96.000	10.000	95.000	85.00	7.000	60.00	1
3	85.76	23.536	90.768	13.920	88.768	96.92	0.768	68.84	3
4	48.12	17.248	97.872	6.496	96.248	72.48	8.248	53.12	0
~~~

> Pengubahan nama kolom ini agar lebih mudah membedakannya dengan kolom `sr` 
(Sleeping Rate). 

5. Melihat informasi dari dataset.  

~~~python
dataset.info()
~~~

**output:**

~~~python

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 630 entries, 0 to 629
Data columns (total 9 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   sr      630 non-null    float64
 1   rr      630 non-null    float64
 2   t       630 non-null    float64
 3   lm      630 non-null    float64
 4   bo      630 non-null    float64
 5   rem     630 non-null    float64
 6   sh      630 non-null    float64
 7   hr      630 non-null    float64
 8   sl      630 non-null    int64  
dtypes: float64(8), int64(1)
memory usage: 44.4 KB
~~~

> Berdasarkan informasi dari datasetnya, data yang ada pada dataset sudah tidak 
memiliki data yang null. Dengan kata lain, Setiap data yang ada sudah terisi semua.

## Pre-processing Data

1. Selanjutnya akan di cek nilai yang unique pada kolom sl (Stress Levels).

~~~ python
dataset['sl'].unique()
~~~

**output:**

~~~ python
array([3, 1, 0, 2, 4])
~~~

>   Berdasarkan output, terdapat 5 nilai yang menjadi hasil prediksi tersebut.
Selanjutnya, ke-5 nilai tersebut akan dikonversi menjadi 3 nilai.

2. Mengubah 5 nilai menjadi yang ada di kolom `sl` menjadi 3 nilai dan menghitung
setiap jumlah setiap nilainya.

~~~ python
qualityType = { 0 : 0, 1 : 0, 2 : 1, 3 : 2, 4 : 2 }
dataset["sl"] =  dataset["sl"].map(qualityType)
dataset.sl.value_counts()
~~~

**output:**

~~~ python
2    252
0    252
1    126
Name: sl, dtype: int64
~~~

> *Note*: 

> - [0, 1] -> 0 = Tidak Stress
> - [2] -> 1 = Stress
> - [3, 4] -> 2 = Sangat Stress

> Perubahan ini saya lakukan karena dengan pertimbangan data yang hanya 630 sehingga
pengkonversian nilai menjadi 3 nilai ini agar akurasinya dapat lebih akurat.

3. Selanjutnya, akan dilakukan pengecekan dataset untuk mengetahui data yang outliers.

~~~ python
fig = px.box(dataset)
fig.show()
~~~

**output:**

![chart](E:\GitHub\Stress-Predict-Project\static\img\plotboxchart.png "Chart")
![chart](https://github.com/MuhammadMukram/Stress-Predict-Project/tree/main/static/img/plotboxchart.png "Chart")

> Dari Box-Plot (Diagram Kotak Garis) di atas tidak diperoleh adanya data outliers 
yang dapat mempengaruhi hasil prediksi. Dengan demikian, data sudah siap diolah.

4. Menentukan kolom yang akan menjadi hasil prediksi.

~~~ python
X = dataset.drop('sl',axis=1).values
y = dataset['sl'].values
~~~

> Variabel y yang merupakan kolom `sl` pada dataset yang digunakan sebagai hasil
prediksi nantinya sebagai hasil yang akan menunjukkan tingkat stress manusia.

**output menampilkan `X`:**
~~~ python
array([[ 93.8  ,  25.68 ,  91.84 , ...,  99.6  ,   1.84 ,  74.2  ],
       [ 91.64 ,  25.104,  91.552, ...,  98.88 ,   1.552,  72.76 ],
       [ 60.   ,  20.   ,  96.   , ...,  85.   ,   7.   ,  60.   ],
       ...,
       [ 97.504,  27.504,  86.88 , ..., 101.88 ,   0.   ,  78.76 ],
       [ 58.64 ,  19.728,  95.728, ...,  84.32 ,   6.728,  59.32 ],
       [ 73.92 ,  21.392,  93.392, ...,  91.96 ,   4.088,  63.48 ]])
~~~

**output menampilkan `y`:**
~~~ python
array([2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1,
       1, 2, 2, 0, 1, 0, 2, 1, 0, 2, 0, 2, 1, 2, 0, 2, 0, 2, 1, 1, 1, 0,
       1, 2, 1, 2, 0, 2, 0, 0, 2, 0, 0, 1, 2, 2, 2, 0, 2, 2, 0, 1, 2, 0,
       0, 1, 2, 0, 0, 2, 0, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0, 2, 1, 0, 0, 0,
       2, 2, 1, 2, 1, 2, 0, 1, 2, 1, 1, 2, 2, 2, 0, 2, 0, 1, 1, 1, 2, 0,
       1, 2, 0, 2, 1, 2, 1, 1, 0, 0, 0, 2, 2, 0, 2, 2, 2, 1, 0, 2, 1, 0,
       2, 2, 0, 0, 0, 1, 2, 0, 1, 2, 1, 0, 1, 0, 0, 0, 0, 0, 2, 0, 2, 2,
       0, 1, 2, 1, 2, 2, 2, 1, 0, 1, 2, 1, 0, 0, 2, 2, 0, 1, 1, 0, 0, 0,
       0, 2, 2, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 2, 2, 2, 0, 0, 0,
       1, 2, 0, 0, 0, 1, 2, 0, 1, 2, 0, 0, 0, 2, 0, 0, 2, 2, 0, 1, 1, 1,
       0, 2, 0, 0, 2, 2, 1, 0, 2, 0, 2, 2, 0, 2, 0, 0, 2, 2, 2, 0, 2, 2,
       2, 2, 2, 0, 2, 1, 2, 2, 0, 2, 1, 2, 0, 2, 1, 2, 2, 1, 2, 2, 1, 0,
       0, 0, 1, 2, 1, 1, 2, 2, 0, 0, 2, 0, 1, 2, 2, 2, 2, 2, 0, 2, 0, 0,
       2, 2, 2, 1, 2, 0, 2, 0, 2, 0, 1, 0, 1, 0, 2, 0, 0, 2, 1, 1, 2, 1,
       2, 0, 2, 1, 0, 0, 0, 0, 2, 2, 2, 0, 2, 2, 1, 2, 2, 2, 0, 1, 0, 1,
       2, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 0, 1, 2, 1, 2, 0, 0, 2, 2, 2, 2,
       2, 2, 2, 1, 0, 2, 1, 2, 1, 0, 1, 1, 2, 0, 2, 0, 0, 2, 2, 2, 1, 0,
       2, 0, 1, 2, 0, 0, 2, 0, 1, 0, 2, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0,
       0, 2, 2, 2, 2, 2, 2, 1, 1, 2, 0, 2, 2, 1, 0, 2, 2, 0, 0, 0, 0, 2,
       2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 2, 1, 0, 1, 2, 1, 1, 2, 2, 0,
       2, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 1, 2, 0,
       2, 1, 1, 0, 2, 2, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 1,
       1, 2, 0, 2, 1, 2, 1, 2, 1, 0, 2, 0, 0, 0, 0, 2, 1, 2, 2, 0, 0, 2,
       1, 2, 2, 0, 1, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 0, 2, 0, 1, 2,
       2, 1, 2, 0, 2, 0, 0, 0, 0, 2, 1, 2, 0, 0, 2, 2, 2, 2, 1, 2, 1, 2,
       0, 2, 1, 0, 1, 2, 1, 2, 2, 0, 0, 0, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2,
       0, 0, 0, 2, 2, 1, 2, 2, 0, 1, 2, 0, 2, 1, 0, 0, 1, 0, 2, 0, 0, 0,
       2, 1, 2, 0, 2, 2, 2, 0, 0, 0, 0, 2, 2, 1, 0, 0, 2, 2, 2, 1, 1, 2,
       0, 1, 0, 0, 2, 0, 1, 2, 0, 1, 0, 2, 0, 1])
~~~

5. Membagi dataset menjadi data training dan data testing.

~~~ python
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.25, random_state=4)
~~~

> Setelah melakukan perintah di atas, data akan dibagi menjadi data testing dan 
data training. Adapun pembagiannya yaitu:
> - Data training (75%)
> - Data testing (25%)

## Pembuatan model

Pada model kali ini akan menggunakan 2 algoritma, yaitu Decision Tree dan 
Support Vector Machines (SVM). Decision Tree cocok digunakan dalam kasus ini 
karena dataset yang digunakan berupa variabel diskrit (datanya dihitung dalam 
waktu terbatas). Adapun algoritma SVM juga cocok digunakan dalam kasus ini karena
dapat menghasilkan model yang baik walaupun himpunan data yang digunakan relatif 
sedikit seperti pada kasus ini.

### Model Decision Tree

1.  Membuat model Decision Tree pada program.
~~~ python
from sklearn.tree import DecisionTreeClassifier
stressTree = DecisionTreeClassifier(criterion="gini", max_depth = 10)
stressTree.fit(X_trainset, y_trainset)
predTree = stressTree.predict(X_testset)
~~~

2. Mengevaluasi akurasi dari model.

~~~ python
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree)*100, '%')
~~~

**output:**

~~~ python
DecisionTrees's Accuracy:  99.36708860759494 %
~~~

> Dapat dilihat bahwa akurasi dari model ini adalah 99.36708860759494 %. Akurasi 
model tersebut menunjukkan angka yang sangat tinggi, hampir sempurna. Hal ini 
menunjukkan bahwa Decision Tree dapat memprediksi dengan akurat.

### Model SVM

1.  Membuat model SVM pada program.
~~~ python
from sklearn import svm
clf = svm.SVC(kernel='linear', gamma='auto').fit(X_trainset,y_trainset) 
y_predict_scm = clf.predict(X_testset)
~~~

2. Mengevaluasi akurasi dari model.

~~~ python
print("SVM's Accuracy: ", round(metrics.accuracy_score(y_testset, y_predict_scm)*100,2),"%")
~~~

**output:**

~~~ python
SVM's Accuracy:  100.0 %
~~~

> Dapat dilihat bahwa akurasi dari model ini adalah 100 %. Akurasi 
model tersebut menunjukkan angka sempurna dalam melakukan prediksi. Hal ini berarti
bahwa model dengan algoritma SVM sangat cocok digunakan dalam kasus ini karena 
sangat akurat dalam melakukan prediksi.