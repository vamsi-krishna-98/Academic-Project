from flask import Flask, render_template
import random
import matplotlib.pyplot as plt
from matplotlib import image
import os
import numpy as np
import keras
import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json
app = Flask(__name__)
city_list = ['0', '1', '#', '+']
key1=random.choice(city_list)
key2=random.choice(city_list)

@app.route('/')
def index():
  return render_template('practice.html',a=key1 ,b=key2)

@app.route('/my-link/')
def my_link():
  print ('I got clicked!')

  #return 'I get Click.'
  def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if (norm==0):
        norm= np.finfo(v.dtype).eps
    return v/norm
  

#loading the model
  model = tf.keras.models.load_model('D:/flask_app/symbol_final_model.h5')
#getting the canvas image
  imgs = os.listdir('D:/flask_app/static/')
  recent = len(imgs)
  img = image.imread('D:/flask_app/static/'+imgs[recent-4])
  

#cropping the canvas image
  y=0
  x=0
  h=100
  w=100
  print(key1)
  print(key2)
  for i in range (0,2):
      x=i*w
      crop = img[y:y+h, x:x+w]
      gray= cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
      resize = 255- cv2.resize(gray, ( 84, 84), interpolation=cv2.INTER_AREA)
      nimg = normalize(np.reshape(resize, (84, 84) , order='C'))
      plt.imshow(nimg , cmap='gray')
      plt.show()
    #predicting the cropped images
      y  = np.argmax(model.predict(nimg.reshape(1, 84, 84, 1)))
      list1 = ["0" , "1" , "#" , "+"]
      if i==0:
          out1=list1[y]
      if i==1:
          out2=list1[y]
          if key2==out1 and key1==out2:
              return "SUCCESSFUL"
          else :
              return "UNSUCCESSFUL"
    #list2=list(tuple)
    #list2.append(list1[y])
    #tuple = tuple(list2)
    #t=liat(tuple)
    #print(list[y], end = ' ')
    #z=list[y]
    #out.append(list[y])
    #z=z+1
    #print(z)
  
  

if __name__ == '__main__':
  app.run(debug=True)


  
  #function for normalizing the image
