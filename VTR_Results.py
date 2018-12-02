from keras.models import model_from_json
import numpy as np
import csv
import math

model = model_from_json(open('model.json').read())
model.load_weights('weights.h5')
data_dir = ""
X_test = np.load(data_dir+'VTR_test_X.npy')
Y = np.load(data_dir+'VTR_test_Y.npy')

names = Y[:, :1]
Y_test = Y[:,1:]
predictions = []

loss1 = 0.0
loss2 = 0.0
loss3 = 0.0
loss4 = 0.0
max_1 = 0.0
max_2 = 0.0
max_3 = 0.0
max_4 = 0.0
list_1 = []
list_2 = []
list_3 = []
list_4 = []
male = [0.0, 0.0, 0.0, 0.0, 0.0, [], [], [], []]
female = [0.0, 0.0, 0.0, 0.0, 0.0, [], [], [], []]
karma_list = [0, 0.0, 0.0, 0.0, 0.0]
AVG_list = [0, 0.0, 0.0, 0.0, 0.0]

y_hat = model.predict(X_test)
for i in range(0,len(Y_test)):
    l1 = np.abs(float(Y_test[i, 0]) - y_hat[i, 0])
    l2 = np.abs(float(Y_test[i, 1]) - y_hat[i, 1])
    l3 = np.abs(float(Y_test[i, 2]) - y_hat[i, 2])
    l4 = np.abs(float(Y_test[i, 3]) - y_hat[i, 3])
    pred = [names[i][0], float(Y_test[i, 0]), float(Y_test[i, 1]), float(Y_test[i, 2]), float(Y_test[i, 3])]

    AVG_list[0] += 1
    AVG_list[1] += float(Y_test[i, 0]) - y_hat[i, 0]
    AVG_list[2] += float(Y_test[i, 1]) - y_hat[i, 1]
    AVG_list[3] += float(Y_test[i, 2]) - y_hat[i, 2]
    AVG_list[4] += float(Y_test[i, 3]) - y_hat[i, 3]

    pred.extend([y_hat[i, 0], y_hat[i, 1], y_hat[i, 2], y_hat[i, 3]])

    if names[i][0].split('_')[3][0] == 'f':
        female[0] += 1
        female[1] += l1
        female[2] += l2
        female[3] += l3
        female[4] += l4
        female[5].append(l1)
        female[6].append(l2)
        female[7].append(l3)
        female[8].append(l4)
    elif names[i][0].split('_')[3][0] == 'm':
        male[0] += 1
        male[1] += l1
        male[2] += l2
        male[3] += l3
        male[4] += l4
        male[5].append(l1)
        male[6].append(l2)
        male[7].append(l3)
        male[8].append(l4)

    predictions.append(pred)

    list_1.append(l1)
    list_2.append(l2)
    list_3.append(l3)
    list_4.append(l4)
    max_1 = max(max_1,l1)
    max_2 = max(max_2,l2)
    max_3 = max(max_3,l3)
    max_4 = max(max_4,l4)

    loss1 += l1
    loss2 += l2
    loss3 += l3
    loss4 += l4

    karma_list[0] += 1
    karma_list[1] += l1 * l1
    karma_list[2] += l2 * l2
    karma_list[3] += l3 * l3
    karma_list[4] += l4 * l4
loss1 /= len(Y_test)
loss2 /= len(Y_test)
loss3 /= len(Y_test)
loss4 /= len(Y_test)
total_loss = loss1+loss2+loss3+loss4
total_loss /= 4.0
print('standard deviation', round(np.std(list_1)*1000, 2), round(np.std(list_2)*1000, 2), round(np.std(list_3)*1000, 2), round(np.std(list_4)*1000, 2))
print('median', round(np.median(list_1)*1000, 2), round(np.median(list_2)*1000, 2), round(np.median(list_3)*1000, 2), round(np.median(list_4)*1000, 2))
print('max loss ', round(max_1*1000, 2), round(max_2*1000, 2), round(max_3*1000, 2), round(max_4*1000, 2))
print('total loss ', round(total_loss*1000, 2))
print('Real test score:', round(loss1*1000, 2), round(loss2*1000, 2), round(loss3*1000, 2), round(loss4*1000, 2))

female[1] = round((female[1] / female[0])*1000, 2)
female[2] = round((female[2] / female[0])*1000, 2)
female[3] = round((female[3] / female[0])*1000, 2)
female[4] = round((female[4] / female[0])*1000, 2)
female[5] = round(np.std(female[5])*1000, 2)
female[6] = round(np.std(female[6])*1000, 2)
female[7] = round(np.std(female[7])*1000, 2)
female[8] = round(np.std(female[8])*1000, 2)

male[1] = round((male[1] / male[0])*1000, 2)
male[2] = round((male[2] / male[0])*1000, 2)
male[3] = round((male[3] / male[0])*1000, 2)
male[4] = round((male[4] / male[0])*1000, 2)
male[5] = round(np.std(male[5])*1000, 2)
male[6] = round(np.std(male[6])*1000, 2)
male[7] = round(np.std(male[7])*1000, 2)
male[8] = round(np.std(male[8])*1000, 2)

print("male: ", male)
print("female: ", female)

# karma

karma_list[1] /= karma_list[0]
karma_list[2] /= karma_list[0]
karma_list[3] /= karma_list[0]
karma_list[4] /= karma_list[0]
print('root mean squared error ', round(math.sqrt(karma_list[1]) * 1000, 2), round(math.sqrt(karma_list[2]) * 1000, 2),
                            round(math.sqrt(karma_list[3]) * 1000, 2), round(math.sqrt(karma_list[4]) * 1000, 2))

AVG_list[1] /= AVG_list[0]
AVG_list[2] /= AVG_list[0]
AVG_list[3] /= AVG_list[0]
AVG_list[4] /= AVG_list[0]
print('AVG ', round(AVG_list[1] * 1000, 2), round(AVG_list[2] * 1000, 2), round(AVG_list[3] * 1000, 2), round(AVG_list[4] * 1000, 2))


with open("results/VTR.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(predictions)
