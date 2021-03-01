from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
import random
import sys
import time
from keras import models


dir_path = "C:\\Users\\Tadeas\\Downloads\\chatbot2\\Intent.json"

f = open(dir_path)

m = models.load_model("chatbotmasta.h5")

everything = json.load(f)

answers = {0: [], 1: [] ,2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: []}

for i in range(22):
  for x in everything['intents'][i]["responses"]:
    answers[i].append(x)

print("If u want to end the chat enter: STOP")

def to_sequnce(mess):
  tokenizer = Tokenizer(num_words= len(mess))
  tokenizer.fit_on_texts(mess)
  seq = tokenizer.texts_to_sequences(mess)
  data = pad_sequences(seq, maxlen= 30)
  return data


while True:
  i = input("You: ")
  if i.upper() == "STOP":
    sys.exit("Robot: Bye")

  else:
    mess = to_sequnce(i)
    pred = m.predict(mess)
    l = np.argmax(pred[0])
    answer = answers[l]
    p = random.randint(0, 5)
    print("writing.....")
    time.sleep(p)
    answer_random = random.randint(0, len(answer)-1)
    answer = answer[answer_random]
    print("Robot: "+ answer)



  
  