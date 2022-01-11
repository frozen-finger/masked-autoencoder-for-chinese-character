import random
import os

file_listi = os.listdir("Fontimage/inference")
file_listc = os.listdir("Fontimage/chinese")
print(len(file_listc))
print(len(file_listi))

cran = random.sample(range(40000), 36000)
iran = random.sample(range(12000), 9000)
for i in cran:
    with open('Fontimage/train/train.txt', 'a', encoding='utf-8') as f:
        f.write('Fontimage/chinese/'+file_listc[i]+"\t"+'0'+'\n')
for i in iran:
    with open('Fontimage/train/train.txt', 'a', encoding='utf-8') as f:
        f.write('Fontimage/inference/' + file_listi[i] + "\t" + '1'+'\n')
for i in range(40000):
    if i not in cran:
        with open('Fontimage/test/test.txt', 'a', encoding='utf-8') as f:
            f.write('Fontimage/chinese/' + file_listc[i] + "\t" + '0'+'\n')
for i in range(12000):
    if i not in iran:
        with open('Fontimage/test/test.txt', 'a', encoding='utf-8') as f:
            f.write('Fontimage/inference/' + file_listi[i] + "\t" + '1'+'\n')