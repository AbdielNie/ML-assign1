# ML CS6375 Assignment1
## For running Part1 code:

If you don't have anaconda you need to open terminal and type:
```
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install statsmodels
pip install logging
pip install sklearn
```
otherwise
```
anaconda install pandas
anaconda install numpy
anaconda install matplotlib
anaconda install seaborn
anaconda install statsmodels
anaconda install logging
anaconda install sklearn
```
Then
```
python part1.py
```
You will see the output print in the terminal:(those are part of )
```commandline
There are 1599 rows and 12 columns
      fixed acidity  volatile acidity  citric acid  ...  sulphates  alcohol  quality
1166            9.9             0.540         0.26  ...       0.98     10.2        5
353            13.5             0.530         0.79  ...       0.77     13.0        5
1079            7.9             0.300         0.68  ...       0.51     12.3        7
326            11.6             0.530         0.66  ...       0.74     11.5        7
916             5.3             0.715         0.19  ...       0.61     11.0        5
1582            6.1             0.715         0.10  ...       0.50     11.9        5
442            15.6             0.685         0.76  ...       0.68     11.2        7
801             8.6             0.550         0.09  ...       0.44     10.0        5
401             7.7             0.260         0.30  ...       0.47     10.8        6
1376            8.2             0.885         0.20  ...       0.46     10.0        5

[10 rows x 12 columns]
Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality'],
      dtype='object')
..............
```
And it will generate a file called:Part1.log
In the log file you will see
the performance of different setting and on the end of log file you will see the best setting of model
```commandline
INFO:root:Epoch 0, Training MSE: 0.17834022989972526
INFO:root:Epoch 100, Training MSE: 0.17332448549246812
INFO:root:Epoch 200, Training MSE: 0.17274170613369855
INFO:root:Epoch 300, Training MSE: 0.17171947944731336
INFO:root:Epoch 400, Training MSE: 0.1726971411064208
INFO:root:Epoch 500, Training MSE: 0.17506056202119952
INFO:root:Epoch 600, Training MSE: 0.177304091503943
INFO:root:Epoch 700, Training MSE: 0.1741238085113747
INFO:root:Epoch 800, Training MSE: 0.1726499560017463
INFO:root:Epoch 900, Training MSE: 0.17219100629164555
INFO:root:Trial with learning_rate=0.01, epochs=1000, Test MSE=0.2647058823529412
INFO:root:Classification report:
              precision    recall  f1-score   support

           0       0.66      0.76      0.70       113
           1       0.81      0.72      0.76       159

    accuracy                           0.74       272
   macro avg       0.73      0.74      0.73       272
weighted avg       0.75      0.74      0.74       272
············
INFO:root:Best settings found: learning_rate=0.5, epochs=1500, MSE=0.0018399264029438822
INFO:root:Classification report for best settings:
              precision    recall  f1-score   support

       False       1.00      1.00      1.00       567
        True       1.00      1.00      1.00       520

    accuracy                           1.00      1087
   macro avg       1.00      1.00      1.00      1087
weighted avg       1.00      1.00      1.00      1087
```

## Part2:

Same with part 1 if you have anaconda do:
```commandline
conda install numpy
conda install pandas 
conda install matplotlib
conda install seaborn
conda install statsmodels
conda install warnings
conda install sklearn
conda install logging
```
otherwise please use pip install