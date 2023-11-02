import sys
import keras
import nltk
import pickle
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
import random
import datetime
import webbrowser
import requests
import billboard
import time
from pygame import mixer

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

words=[]
classes=[]
documents=[]
ignore=['?','!',',',"'s"]

data_file=open('intents.json',encoding="utf-8").read()
intents=json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words=[lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore]
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))
pickle.dump(words,open('model/words.pkl','wb'))
pickle.dump(classes,open('model/classes.pkl','wb'))

#training data
training=[]
output_empty=[0]*len(classes)

for doc in documents:
    bag=[]
    pattern=doc[0]
    pattern=[ lemmatizer.lemmatize(word.lower()) for word in pattern ]
    
    for word in words:
        if word in pattern:
            bag.append(1)
        else:
            bag.append(0)
    output_row=list(output_empty)
    output_row[classes.index(doc[1])]=1
    
    training.append([bag,output_row])
    
random.shuffle(training)
training=np.array(training)  
X_train=list(training[:,0])
y_train=list(training[:,1])  

#Model
model=Sequential()
model.add(Dense(128,activation='relu',input_shape=(len(X_train[0]),)))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation='softmax'))

adam=keras.optimizers.Adam(0.001)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit(np.array(X_train),np.array(y_train),epochs=200,batch_size=10,verbose=1)
weights=model.fit(np.array(X_train),np.array(y_train),epochs=200,batch_size=10,verbose=1)    
model.save('model/mymodel.h5',weights)

from keras.models import load_model
model = load_model('model/mymodel.h5')
intents = json.loads(open('intents.json',encoding="utf-8").read())
words = pickle.load(open('model/words.pkl','rb'))
classes = pickle.load(open('model/classes.pkl','rb'))

#Bỏ comment bên dưới nếu muốn test chạy trực tiếp.
#Chức năng & dự đoán.
# def clean_up(sentence):
#     sentence_words=nltk.word_tokenize(sentence)
#     sentence_words=[ lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def create_bow(sentence,words):
#     sentence_words=clean_up(sentence)
#     bag=list(np.zeros(len(words)))
    
#     for s in sentence_words:
#         for i,w in enumerate(words):
#             if w == s: 
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence,model):
#     p=create_bow(sentence,words)
#     res=model.predict(np.array([p]))[0]
#     threshold=0.8
#     results=[[i,r] for i,r in enumerate(res) if r>threshold]
#     results.sort(key=lambda x: x[1],reverse=True)
#     return_list=[]
    
#     for result in results:
#         return_list.append({'intent':classes[result[0]],'prob':str(result[1])})
#     return return_list

# def get_response(return_list,intents_json):
    
    # if len(return_list)==0:
    #     tag='noanswer'
    # else:    
    #     tag=return_list[0]['intent']
    # if tag=='datetime':        
    #     print(time.strftime("%A"))
    #     print (time.strftime("%d %B %Y"))
    #     print (time.strftime("%H:%M:%S"))

    # if tag=='google':
    #     query=input('Enter query...')
    #     chrome_path = "C:\Program Files\Google\Chrome\Application\chrome.exe%s"
    #     for url in search(query, tld="co.in", num=1, stop = 1, pause = 2):
    #         webbrowser.open("https://google.com/search?q=%s" % query)
    
    # if tag == 'weather':
    #     api_key = 'f756e57e80ff42a88e9445d48c46a597'
    #     base_url = "https://api.openweathermap.org/data/2.5/weather?q"
    #     city_name = input("Enter city name: ")
    #     complete_url = f"{base_url}={city_name}&appid={api_key}"
    
    #     response = requests.get(complete_url)
    
    #     if response.status_code == 200:
    #         data = response.json()
        
    #         if 'main' in data:
    #             temperature = data['main'].get('temp')
    #             feels_like = data['main'].get('feels_like')
    #             weather_description = data['weather'][0]['main']
            
    #             if temperature is not None and feels_like is not None:
    #                 temperature_celsius = round(temperature - 273.15, 2)
    #                 feels_like_celsius = round(feels_like - 273.15, 2)
                
    #                 print(f'Nhiệt độ hiện tại: {temperature_celsius} Celsius')
    #                 print(f'Sẽ cảm giác như: {feels_like_celsius} Celsius')
    #                 print(f'Thời tiết: {weather_description}')
    #             else:
    #                 print('Nhiệt độ không khả dụng')
    #         else:
    #             print('Thông tin thời tiết không khả dụng')
    #     else:
    #         print(f'Error: {response.status_code}')
    # else:
    #     print('Invalid tag.')

        
    # if tag=='news':
    #     main_url = ('http://newsapi.org/v2/everything?'
	# 		'q=Apple&'
	# 		'sortBy=popularity&'
	# 		'apiKey=6b9f13b21e1b4eec9bd9e0c722f07b74')
    #     open_news_page = requests.get(main_url).json()
    #     article = open_news_page["articles"]
    #     results = [] 
          
    #     for ar in article: 
    #         results.append([ar["title"],ar["url"]]) 
          
    #     for i in range(10): 
    #         print(i + 1, results[i][0])
    #         print(results[i][1],'\n')
            
    
    # if tag=='song':
    #     chart=billboard.ChartData('hot-100')
    #     print('The top 10 songs at the moment are:')
    #     for i in range(10):
    #         song=chart[i]
    #         print(song.title,'- ',song.artist)
    # if tag=='timer':        
    #     mixer.init()
    #     x=input('Minutes to timer..')
    #     time.sleep(float(x)*60)
    #     mixer.music.load('Handbell-ringing-sound-effect.mp3')
    #     mixer.music.play()
        
    # list_of_intents= intents_json['intents']    
    # for i in list_of_intents:
    #     if tag==i['tag'] :
    #         result= random.choice(i['responses'])
    # return result

# def response(text):
    # return_list=predict_class(text,model)
    # response=get_response(return_list,intents)
    # return response

# while(1):
    # x=input()
    # print(response(x))
    # if x.lower() in ['bye','goodbye','get lost','see you']:  
    #     break


#Self learning
print('Giúp tôi học thêm?')
tag=input('Nhập danh mục chung của câu hỏi hoặc nhập E để thoát: ')
flag=-1
if tag in ('E','e'):
    sys.exit()
else:
    for i in range(len(intents['intents'])):
        if tag.lower() in intents['intents'][i]['tag']:
            intents['intents'][i]['patterns'].append(input('Enter your message: '))
            intents['intents'][i]['responses'].append(input('Enter expected reply: '))        
            flag=1

    if flag==-1:
    
        intents['intents'].append (
            {'tag':tag,
            'patterns': [input('Please enter your message')],
            'responses': [input('Enter expected reply')]})
    
    with open('intents.json','w',encoding="utf-8") as outfile:
        outfile.write(json.dumps(intents,indent=4))