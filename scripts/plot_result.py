import numpy as np
import matplotlib.pyplot as plt

model1 = {'valid_F1': [], 'test_F1': []}
model2 = {'valid_F1': [], 'test_F1': []}
model3 = {'valid_F1': [], 'test_F1': []}
model4 = {'valid_F1': [], 'test_F1': []}
result = [model1, model2, model3, model4]


with open('../step:100_epochs:40.out', 'r') as f:
    result_count = 0
    lines = f.readlines()
    for line in lines:
        words = line.split()
        if (len(words) > 0) and (words[0] == 'BEST'):
            valid_F1 = float(words[7])
            test_F1 = float(words[10])
            result[result_count % 4]['valid_F1'].append(valid_F1)
            result[result_count % 4]['test_F1'].append(test_F1)
            result_count += 1
t = np.arange(100, 3700, 100)

# print(map(int, result[0]['valid_F1'][0:4]))
for i in range(4, 34, 5):
    print(result[2]['test_F1'][i])

plt.figure(1)
line1, line2, line3, line4 = plt.plot(t, result[0]['valid_F1'][0:36], 'r-^',
                                      t, result[1]['valid_F1'][0:36], 'g-o',
                                      t, result[2]['valid_F1'][0:36], 'b-s',
                                      t, result[3]['valid_F1'][0:36], 'y-*',
                                      linewidth=2.0, markersize=20)
plt.xticks(np.arange(100, 3700, 500), fontsize=30)
plt.yticks(np.arange(70, 100, 5), fontsize=30)
plt.xlabel('Data set size (number of sentences)', fontsize=50)
plt.ylabel('Valid F1 (%)', fontsize=50)
plt.legend((line1, line2, line3, line4),
           ('Baseline LSTM', 'LSTM + LM', 'Bi-LSTM + LM', 'Bi-LSTM + LM + GloVe'),
           fontsize=50,
           loc='lower right')


plt.figure(2)
line1, line2, line3, line4 = plt.plot(t, result[0]['test_F1'][0:36], 'r-^',
                                      t, result[1]['test_F1'][0:36], 'g-o',
                                      t, result[2]['test_F1'][0:36], 'b-s',
                                      t, result[3]['test_F1'][0:36], 'y-*',
                                      linewidth=2.0, markersize=20)
plt.xticks(np.arange(100, 3700, 500), fontsize=30)
plt.yticks(np.arange(70, 100, 5), fontsize=30)
plt.xlabel('Data set size (number of sentences)', fontsize=50)
plt.ylabel('Test F1 (%)', fontsize=50)
plt.legend((line1, line2, line3, line4),
           ('Baseline LSTM', 'LSTM + LM', 'Bi-LSTM + LM', 'Bi-LSTM + LM + GloVe'),
           fontsize=50,
           loc='lower right')
plt.show()
