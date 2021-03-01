import json
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers
from keras import optimizers
import os

dir_path = "C:\\Users\\Tadeas\\Downloads\\chatbot2\\Intent.json"

f = open(dir_path)

everything = json.load(f)

text = []
label = []

for i in range(22):
  for x in everything['intents'][i]["text"]: 
    text.append(x)
    label.append(i)

answers = {0: [], 1: [] ,2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: []}

for i in range(22):
  for x in everything['intents'][i]["responses"]:
    answers[i].append(x)

labels = np.asarray(label, dtype="float32")

labels = to_categorical(labels, num_classes=22, dtype="float32")


tokenizer = Tokenizer(num_words=30)
tokenizer.fit_on_texts(text)
seq = tokenizer.texts_to_sequences(text)
data = pad_sequences(seq, maxlen= 30)


realsamples = np.arange(data.shape[0])
np.random.shuffle(realsamples)
data = data[realsamples]
labels = labels[realsamples]


glove_dir = "C:\\Users\\Tadeas\\Downloads\\glove.6b"

embeddings_ind = {}
f = open(os.path.join(glove_dir, "glove.6B.100d.txt"), encoding="utf8")
for l in f:
  values = l.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype="float32")
  embeddings_ind[word] = coefs
f.close()

embeding_matrix = np.zeros((len(data), 100))

for word, ind in tokenizer.word_index.items():
  if ind < len(data):
    embeding_vector = embeddings_ind.get(word)
    if embeding_vector is not None:
      embeding_matrix[ind] = embeding_vector

m = models.Sequential()

m.add(layers.Embedding(data.shape[0], 100, input_length= 30))
m.add(layers.Conv1D(64, 7, padding="same", activation="relu"))
m.add(layers.MaxPooling1D(5))
m.add(layers.Conv1D(128, 7, padding="same", activation="relu"))
m.add(layers.MaxPooling1D(3))
m.add(layers.Conv1D(256, 7, padding="same", activation="relu"))
m.add(layers.Conv1D(256, 7, padding="same", activation="relu"))
m.add(layers.MaxPooling1D(1))
m.add(layers.Conv1D(256, 7, padding="same", activation="relu"))
m.add(layers.MaxPooling1D(1))
m.add(layers.Conv1D(256, 7, padding="same", activation="relu"))
m.add(layers.Conv1D(256, 7, padding="same", activation="relu"))
m.add(layers.GlobalMaxPooling1D())

m.add(layers.Dense(22, activation="softmax"))

m.layers[0].set_weights([embeding_matrix])
m.layers[0].trainable = False

m.compile(optimizer=optimizers.Adam(lr=0.001), loss="categorical_crossentropy", metrics=["acc"])

m.fit(data, labels, epochs=50, batch_size=4, validation_split=0.05)
m.save("chatbotmasta3.h5")













    
    
    




