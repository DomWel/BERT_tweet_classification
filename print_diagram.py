from matplotlib import pyplot as plt
import json
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import config

# Load JSON file
with open('/content/drive/MyDrive/BERT/tweet_class/results_4e/training_phase1_hist') as json_file:
    hist1 = json.load(json_file)

with open('/content/drive/MyDrive/BERT/tweet_class/results_4e/training_phase2_hist') as json_file:
    hist2 = json.load(json_file)

font = {'fontname':'Arial'}


print(hist1['acc']+hist2['acc'])
print(hist1['val_acc']+hist2['val_acc'])

plt.plot(hist1['acc']+hist2['acc'])
plt.plot(hist1['val_acc']+hist2['val_acc'])
plt.title('model accuracy')

len_x = len(hist1['acc']+hist2['acc'])
plt.xticks(np.arange(len_x), np.arange(1, len_x+1))
#plt.yticks(np.arange(5), [0.6, 0.7, 0.8, 0.9, 1.0])

plt.ylabel('accuracy', **font)
plt.xlabel('epoch', **font)
plt.legend(['train', 'val'], loc='upper left')
fig = plt.gcf()
plt.show()
plt.draw()
fig.savefig(config.dirs['results_path'] + '/training_diag_acc.png')