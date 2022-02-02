import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd
import os # dosya konumu ile ilgili sıkıntı yaşamamak için dosya konumunu belirticez



path= os.path.dirname(__file__)
os.chdir(path)


def load_mnist():
    with open('mnist.pkl','rb') as f:
        mnist = pickle.load(f)
    
    return mnist['training_images'],mnist['training_labels'],mnist['test_images'],mnist['test_labels']


train_x,train_y,test_x,test_y = load_mnist() #aldığımız 4 değişkeni Train ve Test değerlerine atadık


#4 değişkeni pandasın dataframe yapısına dönüştürdük
train_x,train_y,test_x,test_y = [pd.DataFrame(x) for x in [train_x,train_y,test_x,test_y]]

#normalizasyon işlemini gerçekleştiriyoruz böylelikle tüm değerleri(pikselleri) 0 la 1 arasında boyutlandırdık
train_x=train_x/255.0
test_x=test_x/255.0

svc = SVC() #SVC modelini oluşturduk


svc.fit(train_x,train_y.values.flatten())  #y değerlerini güncelledik flattenla 1 boyutla hale getirdik. Bunun sebebi
                                           #fit fonksiyonu 2. değişken olarak 1 boyutlu değişken kabul ediyor

filename = "svm_model.pkl"
pickle.dump(svc, open(filename,"wb")) 

y_pred= svc.predict(test_x)
print(classification_report(test_y, y_pred))



