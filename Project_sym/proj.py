from flask import Flask, render_template, redirect, url_for, flash
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
symbol_list = ['0', '1', '#', '+']


@app.route('/')
def index():
  index.key1=random.choice(symbol_list)
  index.key2=random.choice(symbol_list)
  return render_template('1_web.html',a=index.key1 ,b=index.key2)

@app.route('/main/')
def main():
  index.key1=random.choice(symbol_list)
  index.key2=random.choice(symbol_list)
  return render_template('1_web.html',a=index.key1 ,b=index.key2)

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
  model = tf.keras.models.load_model('E:/project_sym/Project_sym/symbolsnew.h5',custom_objects={'softmax_v2':tf.nn.softmax})
#getting the canvas image
  imgs = os.listdir('E:/project_sym/Project_sym/static/')
  recent = len(imgs)
  img = image.imread('E:/project_sym/Project_sym/static/'+imgs[recent-6])
  
  

#cropping the canvas image
  y=0
  x=0
  h=100
  w=100
  print(index.key1)
  print(index.key2)
  for i in range (0,2):
      x=i*w
      crop = img[y:y+h, x:x+w]
      gray= cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
      resize = 255- cv2.resize(gray, ( 84, 84), interpolation=cv2.INTER_AREA)
      nimg = normalize(np.reshape(resize, (84, 84) , order='C'))
      plt.imshow(nimg , cmap='gray')
      #plt.show()
    #predicting the cropped images
      y  = np.argmax(model.predict(nimg.reshape(1, 84, 84, 1)))
      list1 = ["0" , "1" , "#" , "+"]
      if i==0:
          out1=list1[y]
      if i==1:
          out2=list1[y]
          if index.key2==out1 and index.key1==out2:
              return render_template('success.html')
          else :
              return render_template('unsuccess.html')
          

    
   
  
  

if __name__ == '__main__':
  app.run(debug=True)


