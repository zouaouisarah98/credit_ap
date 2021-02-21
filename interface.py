from tkinter import *
import tkinter as tk
from PIL import ImageTk,Image
def k_means(): 
 def tf_5():
     #!/usr/bin/python3
     # -*- coding: utf-8 -*-
     import pandas as pd
     import numpy as np
     import matplotlib.pyplot as plt
     import seaborn as sns
     import sklearn
     import os
     import csv
     import sys

     from nltk.probability import FreqDist
     from nltk.corpus import stopwords
     from sklearn.cluster import KMeans
     from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

     stop_words=set(stopwords.words("french"))
     df = pd.read_csv('tok.csv', encoding='latin-1')
     df['text'] = df['text'].str.replace('\n', '')
     test=df.head()
     print(test)
     # covert words into TFIDF metrics
     tfidf = TfidfVectorizer(stop_words)
     X= tfidf.fit_transform(df['text'])
     print(X)
     # number of clusters we want
     k = 5
     model= KMeans(n_clusters= k, init='k-means++', max_iter=100, n_init=1,random_state=0)
     #X_2d = model.fit(X)
     X_2d = model.fit_transform(X)
     print (X_2d)
     print("Top terms oour cluster:")
     order_centroids = model.cluster_centers_.argsort()[:, ::-1]
     terms = tfidf.get_feature_names()
     for i in range(k):
         print("Cluster %d:" % i),
         for ind in order_centroids[i, :10]:
             print(' %s' % terms[ind]),
         print
     print("\n")
     # predict our clusters for each text
     X_clustered = model.fit_predict(X_2d)
     # display by groups
     df_plot = pd.DataFrame(list(X_2d), list(X_clustered))
     df_plot = df_plot.reset_index()
     df_plot.rename(columns = {'index': 'Cluster'}, inplace = True)
     df_plot['Cluster'] = df_plot['Cluster'].astype(int)
     print(df_plot.head())
     #prediction
     print(X_clustered)
     print(df_plot.groupby('Cluster').agg({'Cluster': 'count'}))
     # make a column for name by clusters
     col = df_plot['Cluster'].map({0:'Engagement ', 1:'à l`arrêt', 2: 'Changement d’état  ',3:'dans un carrefour',4:'Perte du contrôle'})
     sns.countplot(x=col, data=df, palette="mako_r") 
     plt.show()
 def tf_4():
    #!/usr/bin/python3
    # -*- coding: utf-8 -*-
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sklearn
    import os
    import csv
    import sys
    from nltk.probability import FreqDist
    from nltk.corpus import stopwords
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    stop_words=set(stopwords.words("french"))
    df = pd.read_csv('tok.csv', encoding='latin-1')
    df['text'] = df['text'].str.replace('\n', '')
    test=df.head()
    print(test)
    # covert words into TFIDF metrics
    tfidf = TfidfVectorizer(stop_words)
    X= tfidf.fit_transform(df['text'])
    print(X)
    # number of clusters we want
    k = 4
    model= KMeans(n_clusters= k, init='k-means++', max_iter=100, n_init=1,random_state=0)
    #X_2d = model.fit(X)
    X_2d = model.fit_transform(X)
    print (X_2d)
    print("Top terms oour cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf.get_feature_names()
    for i in range(k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),
        print
    print("\n")
    # predict our clusters for each text
    X_clustered = model.fit_predict(X_2d)
    # display by groups
    df_plot = pd.DataFrame(list(X_2d), list(X_clustered))
    df_plot = df_plot.reset_index()
    df_plot.rename(columns = {'index': 'Cluster'}, inplace = True)
    df_plot['Cluster'] = df_plot['Cluster'].astype(int)
    print(df_plot.head())
    #prediction
    print(X_clustered)
    print(df_plot.groupby('Cluster').agg({'Cluster': 'count'}))
    # make a column for name by clusters
    col = df_plot['Cluster'].map({0:'changement de direction ', 1:'perte de contrôle ', 2: 'heurté à l arrière ',3:'à l`arrêt'})
    sns.countplot(x=col, data=df, palette="mako_r") 
    plt.show() 
 def w2v_4():
     #!/usr/bin/python3
     # -*- coding: utf-8 -*-
     import pandas as pd 
     import numpy as np
     import matplotlib.pyplot as plt
     import seaborn as sns
     import sklearn
     import os
     import csv
     import sys
   
     from nltk.corpus import stopwords
     from sklearn.cluster import KMeans

     from gensim.models import word2vec
     from gensim.utils import tokenize

     stop_words=set(stopwords.words("french"))
     df = pd.read_csv('tok.csv', encoding='latin-1')
     df['text'] = df['text'].str.replace('\n', '')
     print(df.head())
     sentance = [list(tokenize(s, deacc=True, lower=True)) for s in df['text']]
     model = word2vec.Word2Vec(sentance, size=300, window=20,min_count=2, workers=1, iter=100)
    
     print(model.corpus_count)
     def get_vect(word, model):
         try:
              return model.wv[word]
         except KeyError:
             return np.zeros((model.vector_size,))

     def sum_vectors(phrase, model):
         return sum(get_vect(w, model) for w in phrase)

     def word2vec_features(X, model):
         feats = np.vstack([sum_vectors(p, model) for p in X])
         return feats
    
     wv_train_feat = word2vec_features(sentance, model)
     print(wv_train_feat)
     k = 4
     modell= KMeans(n_clusters= k, init='k-means++', max_iter=100, n_init=1,random_state=0)
  
     X_2d = modell.fit_transform(wv_train_feat)
     print(X_2d)
     # predict our clusters for each text
     X_clustered = modell.fit_predict(X_2d)
     # display by groups
     df_plot = pd.DataFrame(list(X_2d), list(X_clustered))
     df_plot = df_plot.reset_index()
     df_plot.rename(columns = {'index': 'Cluster'}, inplace = True)
     df_plot['Cluster'] = df_plot['Cluster'].astype(int)
     print(df_plot.head())
     #prediction
     print(X_clustered)
     print(df_plot.groupby('Cluster').agg({'Cluster': 'count'}))
     col = df_plot['Cluster'].map({0:' Fausse manœuvre ', 1:' à l`arrêt ', 2: ' Les infraction au code de la route ',3:' heurté à l arrière /perte de contrôle'})
     sns.countplot(x=col, data=df, palette="mako_r") 
     plt.show()   
 def w2v_5():
     #!/usr/bin/python3
     # -*- coding: utf-8 -*-
     import pandas as pd 
     import numpy as np  
     import matplotlib.pyplot as plt
     import seaborn as sns
     import sklearn
     import os
     import csv
     import sys
     from nltk.corpus import stopwords
     from sklearn.cluster import KMeans
     from gensim.models import word2vec
     from gensim.utils import tokenize
     stop_words=set(stopwords.words("french"))
     df = pd.read_csv('tok.csv', encoding='latin-1')
     df['text'] = df['text'].str.replace('\n', '')
     print(df.head())
     sentance = [list(tokenize(s, deacc=True, lower=True)) for s in df['text']]
     
     model = word2vec.Word2Vec(sentance, size=300, window=20,min_count=2, workers=1, iter=100)
     print(model.corpus_count)
     def get_vect(word, model):
         try:
             return model.wv[word]
         except KeyError:
             return np.zeros((model.vector_size,))
 
     def sum_vectors(phrase, model):
         return sum(get_vect(w, model) for w in phrase)

     def word2vec_features(X, model):
         feats = np.vstack([sum_vectors(p, model) for p in X])
         return feats
    
     wv_train_feat = word2vec_features(sentance, model)
     print(wv_train_feat)
     k = 5
     modell= KMeans(n_clusters= k, init='k-means++', max_iter=100, n_init=1,random_state=0) 
     X_2d = modell.fit_transform(wv_train_feat)
     print(X_2d)
     # predict our clusters for each text
     X_clustered = modell.fit_predict(X_2d)
     # display by groups
     df_plot = pd.DataFrame(list(X_2d), list(X_clustered))
     df_plot = df_plot.reset_index()
     df_plot.rename(columns = {'index': 'Cluster'}, inplace = True)
     df_plot['Cluster'] = df_plot['Cluster'].astype(int)
     print(df_plot.head())
     #prediction
     print(X_clustered)
     print(df_plot.groupby('Cluster').agg({'Cluster': 'count'}))
     col = df_plot['Cluster'].map({0:' Fausse manœuvre ', 1:' à l`arrêt ', 2: ' Les infraction au code de la route ',3:' heurté à l arrière', 4:' perte de contrôle'})
     sns.countplot(x=col, data=df, palette="mako_r") 
     plt.show()
 def text():
        dic={}
        from PIL import ImageTk,Image
        window = Toplevel()
        window.title("Les Textes")
        window.geometry("800x500")
        window.minsize(720,480)
        window.maxsize(800,600)
        window.iconbitmap("LG.ico")
        window.config()
        label_titel = Label(window, text= "            Les Textes ", font=("courrier",40),fg="#1DC6C8")
        label_titel.grid(row=1,column=2, sticky=W)
        label_titel = Label(window, text= "      ", font=("courrier",40))
        label_titel.grid(row=2,column=1,sticky=W)
        canvas1=Canvas(window,width=600,height=300)
        image1=ImageTk.PhotoImage(Image.open("fd.png"))
        dic["image1"]=image1
        canvas1.create_image(80,35,anchor=NW,image=image1)
        canvas1.grid(row=2, column=2,sticky=W)
        frame=Frame(window)
        frame.grid(row=3,column=2,sticky=W)
        label1= Label(frame, text= " TF-IDF         ", font=("courrier",30),fg="#566573")
        label1.grid(row=1, column=1, sticky=W)
        label2= Label(frame, text= " Word2Vec", font=("courrier",30),fg="#566573")
        label2.grid(row=1, column=3, sticky=W)
        liste_tf4=["changement de direction ", "perte de contrôle", "heurté à l arrière", "à l`arrêt"]
        def op1(event):
                    import os
                    #my= Label(window,text=clic3.get())
                    if clic3.get() == " à l`arrêt ":
                        #os.chdir("C:/Users/Lenovo/Desktop/hello")
                        os.startfile("w2v5 à l`arrêt.pdf")
                    elif clic3.get() == "Fausse manœuvre ":
                        os.startfile("Fausse manœuvre.pdf")
                    elif clic3.get() == " Les infraction au code de la route ":
                        os.startfile("w2v5 infraction au code de la route.pdf")
                    elif clic3.get() == " heurté à l arrière":
                        os.startfile("w2v5  heurté à l_arrière.pdf")
                    else:
                        os.startfile("w2v5 perte de contrôle.pdf")
        def op3(event):
                    import os
                    #my= Label(window,text=clic1.get())
                    if clic1.get() == "à l`arrêt " :
                        #os.chdir("C:/Users/Lenovo/Desktop/hello")
                        os.startfile("tf5 à l_arrêt.pdf")
                    elif clic1.get() == "Engagement":
                        os.startfile("tf5 engagement.pdf")
                    elif clic1.get() == "changement d`état ":
                        os.startfile("tf5 changement d_état (arrêter_ démarrer ).pdf")
                    elif clic1.get() == "dans un carrefour ":
                        os.startfile("tf5 carrefour.pdf")
                    else:
                        os.startfile("tf5 perte le contrôle.pdf")
        def op2(event):
                    import os
                    #my= Label(window,text=clic2.get())
                    if clic2.get() =="à l`arrêt ":
                        #os.chdir("C:/Users/Lenovo/Desktop/hello")
                        os.startfile("w2v4 à l’arrêt.pdf")
                    elif clic2.get() == "Fausse manœuvre ":
                        os.startfile("Fausse manœuvre.pdf")
                    elif clic2.get() == " Les infraction au code de la route ":
                        os.startfile("w2v4 infraction au code de la route.pdf")
                    else:
                        os.startfile("w2v4 perte de Control _heurté à l arrière.pdf")
        def op(event):
                    import os
                    #my= Label(window,text=clic.get())
                    if clic.get() == "changement de direction ":
                        #os.chdir("C:/Users/Lenovo/Desktop/hello")
                        os.startfile("tf4 changement de direction.pdf")
                    elif clic.get() == "perte de contrôle" :
                        os.startfile("tf4 perte de contrôle.pdf")
                    elif clic.get() == "heurté à l arrière" :
                        os.startfile("tf4  heurté à l arrière.pdf")
                    else:
                        os.startfile("tf4  à l_arrêt.pdf")
        clic= StringVar()
        clic.set(liste_tf4[0])
        drop= OptionMenu(frame,clic,*liste_tf4,command= op)
        drop.grid(row=2, column=1, sticky=W)
        liste_tf5=["Engagement", "à l`arrêt ", "changement d`état ", "dans un carrefour ", "perte de contrôle"]
        liste_W2V4=[ "Fausse manœuvre ", "à l`arrêt "," Les infraction au code de la route "," heurté à l arrière /perte de contrôle"]
        liste_W2V5=["Fausse manœuvre ", " à l`arrêt ", " Les infraction au code de la route ", " heurté à l arrière", " perte de contrôle"]
        label_titel = Label(frame, text= "k=4:      ", font=("courrier",20),fg="#566573")
        label_titel.grid(row=2, column=0, sticky=W)
        label_tite2 = Label(frame, text= "k=5:      ", font=("courrier",20),fg="#566573")
        label_tite2.grid(row=4, column=0, sticky=W)
        clic1= StringVar()
        clic1.set(liste_tf5[0])
        clic2= StringVar()
        clic2.set(liste_W2V4[0])
        clic3= StringVar()
        clic3.set(liste_W2V5[0])
        drop1= OptionMenu(frame,clic1,*liste_tf5,command= op3)
        drop1.grid(row=4, column=1, sticky=W)
        drop2= OptionMenu(frame,clic2,*liste_W2V4,command= op2)
        drop2.grid(row=2, column=3, sticky=W)
        drop3= OptionMenu(frame,clic3,*liste_W2V5,command= op1)
        drop3.grid(row=4, column=3, sticky=W)
        window.mainloop()
 #creation
 dic={}
 window2 = Toplevel()
 window2.title("clustring ")
 window2.geometry("1000x700")
 window2.minsize(720,480)
 window2.maxsize(1000,700)
 window2.iconbitmap("LG.ico")
 window2.config()
 frame=Frame(window2)
 frame1=Frame(frame)
 frame2=Frame(frame)
 
 label_titel = Label(window2, text= "k-maens clustring", font=("courrier",40),bg="#1DC6C8",fg="#34495E")
 label_titel.pack(expand= YES,fill=X,pady=23)
 label1= Label(frame1, text= " TF-IDF :", font=("courrier",30),fg="#1DC6C8")
 label1.pack(expand= YES,pady=35,fill=X)
 frame1.grid(row=0, column=1, sticky=W)
 button1=Button(frame1,text= "K=4", font=("courrier",25),bg="#34495E",fg="white",command=tf_4)
 button1.pack(expand= YES,pady=25,fill=X)
 but1=Button(frame1,text= "k=5", font=("courrier",25),bg="#34495E",fg="white",command=tf_5)
 but1.pack(expand= YES,pady=25,fill=X)
 label2= Label(frame2, text= "Word2Vec :", font=("courrier",30),fg="#1DC6C8")
 label2.pack(expand= YES,pady=35,fill=X)
 frame2.grid(row=0, column=3, sticky=W)
 button2=Button(frame2,text= "K=4", font=("courrier",25),bg="#34495E",fg="white",command=w2v_4)
 button2.pack(expand= YES,pady=25,fill=X)
 but2=Button(frame2,text= "k=5", font=("courrier",25),bg="#34495E",fg="white",command=w2v_5)
 but2.pack(expand= YES,pady=25,fill=X)
 frame.pack(expand = YES)
 frame3=Frame(frame)
 frame3.grid(row=0, column=2, sticky=W)
 but3=Button(frame3,text= "consulter les textes", font=("courrier",25),bg="#34495E",fg="white",command=text)
 but3.pack(expand= YES,pady=25,fill=X)
 canva=Canvas(frame3,width=450,height=450)
 image3=ImageTk.PhotoImage(Image.open("im.png"))
 dic["image3"]=image3
 canva.create_image(150,150,anchor=NW,image=image3)
 canva.pack(expand=YES,fill=X)
 

 menu_bar = Menu(window2)
 file_menu = Menu(menu_bar,tearoff=0)
 file_menu1 = Menu(menu_bar,tearoff=0)
 file_menu.add_command(label= "Ouvrir")
 file_menu.add_command(label= "Quiter", command= window2.quit)
 file_menu1 = Menu(menu_bar,tearoff=0)
 file_menu1.add_command(label= "les résultats")
 file_menu1.add_command(label= "les textes")
 menu_bar.add_cascade(label= "Fichier", menu= file_menu)
 menu_bar.add_cascade(label= "TF-IDF", menu= file_menu1)
 menu_bar.add_cascade(label= "Word2Vec", menu= file_menu1)
 window2.config(menu=menu_bar)








 # affichage
 window2.mainloop()
def prét():
 import tkinter
 import nltk
 import string

 from PIL import ImageTk,Image
 import matplotlib.pyplot as plt

 from nltk.probability import FreqDist
 from nltk.corpus import stopwords
 #lemmitation
 from nltk.stem import WordNetLemmatizer
 from nltk.stem import wordnet
 wnl = WordNetLemmatizer()

 #strimming
 from nltk import stem
 from nltk.stem import PorterStemmer
 from nltk.stem.snowball import FrenchStemmer
 stemmer = FrenchStemmer()

 #POS
 from nltk.tag.stanford import StanfordPOSTagger
 from nltk.tag import UnigramTagger
 #from nltk import averaged_perceptron_tagger
 from nltk.tag import pos_tag 

 #tokinization
 from nltk.tokenize import word_tokenize
 from nltk.tokenize import sent_tokenize
 from nltk.tokenize.punkt import PunktTrainer
 #def prédect(): 
 #creation

 dic={}
 window = Toplevel()
 window.title("Prétraitement")
 window.geometry("1080x650")
 window.minsize(480,360)
 window.maxsize(1080,650)
 window.iconbitmap("LG.ico")
 window.config()
 frame=Frame(window)
 frame1=Frame(frame)#bd=1,relief =SUNKEN
 frame2=Frame(frame)
 def txt():
      root = Tk()
      root.title("Texte")
      root.geometry("800x150")
      #root.minsize(480,360)
      root.maxsize(800,150)
      root.iconbitmap("LG.ico")
      root.config()
      T1="""Étant à l'arrêt au feu tricolore (rouge) j'ai été percuté à l'arrière par
         le véhicule B, son conducteur n'ayant pas réussi à s'arrêter avant mon véhicule.
         Mon véhicule a subi des dégâts visuels à l'arrière. Sous réserve d'autres dégâts."""
      leb=Label(root ,text=T1, font=("courrier",14)).pack(expand=YES) 
      root.mainloop()
 def net():
      root = Tk()
      root.title("Nettoyage")
      root.geometry("800x150")
      #root.minsize(480,360)
      root.maxsize(800,150)
      root.iconbitmap("LG.ico")
      root.config()
      text=""" Étant à l'arrêt au feu tricolore (rouge) j'ai été percuté à l'arrière par
         le véhicule B, son conducteur n'ayant pas réussi à s'arrêter avant mon véhicule.
         Mon véhicule a subi des dégâts visuels à l'arrière. Sous réserve d'autres dégâts."""
      # Removing puntuation
      translator_1 = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
      text= text.translate(translator_1)
      # Removing numbers
      text= re.sub(r'\d+', ' ',text)
      bad_chars = [' b ',' B ',' c ',' C ',' d ',' D ',' e ',' E ',' f ',' F ',' g ',' G ',' h ',' H ',' a ', ' A ',' i ',' I ',' j ',' J ',' K ',' k ',' l ',' L ',' m ',' M ',' n ',' N ',' o ',' O ',' p ',' P ',' q ',' Q ',' r ',' R ',' s ',' S ',' t ',' T ',' u ',' U ',' v ',' V ',' w ',' W ',' x ',' X ',' y ',' Y ',' z ',' Z' "*"] 
      for i in bad_chars : 
          text= re.sub(i , ' ',text)
      #print(text)
      leb=Label(root ,text=text, font=("courrier",14)).pack(expand=YES) 
      root.mainloop()
 def tok():
      root = Tk()
      root.title("Les Tokens")
      root.geometry("1200x80")
      #root.minsize(480,360)
      root.maxsize(1200,80)
      root.iconbitmap("LG.ico")
      root.config()
      text="""Étant à l'arrêt au feu tricolore (rouge) j'ai été percuté à l'arrière par
         le véhicule B, son conducteur n'ayant pas réussi à s'arrêter avant mon véhicule.
         Mon véhicule a subi des dégâts visuels à l'arrière. Sous réserve d'autres dégâts."""
      # Removing puntuation
      translator_1 = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
      text= text.translate(translator_1)
      # Removing numbers
      text= re.sub(r'\d+', ' ',text)
      bad_chars = [' b ',' B ',' c ',' C ',' d ',' D ',' e ',' E ',' f ',' F ',' g ',' G ',' h ',' H ',' a ', ' A ',' i ',' I ',' j ',' J ',' K ',' k ',' l ',' L ',' m ',' M ',' n ',' N ',' o ',' O ',' p ',' P ',' q ',' Q ',' r ',' R ',' s ',' S ',' t ',' T ',' u ',' U ',' v ',' V ',' w ',' W ',' x ',' X ',' y ',' Y ',' z ',' Z' "*"] 
      for i in bad_chars : 
          text= re.sub(i , ' ',text)
      #print(text)
      tokenized_word=nltk.word_tokenize(text)
      stop_words=set(stopwords.words("french"))
      #Remove stop words
      filtered_sent=[]
      for w in tokenized_word:
          if w not in stop_words:
              filtered_sent.append(w)
      leb=Label(root ,text=filtered_sent, font=("courrier",12)).pack(expand=YES) 
      root.mainloop()
 def lem():
      root = Tk()
      root.title("Lemmatisation")
      root.geometry("1200x80")
      #root.minsize(480,360)
      root.maxsize(1200,80)
      root.iconbitmap("LG.ico")
      root.config()
      text="""Étant à l'arrêt au feu tricolore (rouge) j'ai été percuté à l'arrière par
         le véhicule B, son conducteur n'ayant pas réussi à s'arrêter avant mon véhicule.
         Mon véhicule a subi des dégâts visuels à l'arrière. Sous réserve d'autres dégâts."""
     # Removing puntuation
      translator_1 = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
      text= text.translate(translator_1)
      # Removing numbers
      text= re.sub(r'\d+', ' ',text)
      bad_chars = [' b ',' B ',' c ',' C ',' d ',' D ',' e ',' E ',' f ',' F ',' g ',' G ',' h ',' H ',' a ', ' A ',' i ',' I ',' j ',' J ',' K ',' k ',' l ',' L ',' m ',' M ',' n ',' N ',' o ',' O ',' p ',' P ',' q ',' Q ',' r ',' R ',' s ',' S ',' t ',' T ',' u ',' U ',' v ',' V ',' w ',' W ',' x ',' X ',' y ',' Y ',' z ',' Z' "*"] 
      for i in bad_chars : 
          text= re.sub(i , ' ',text)
      #print(text)
      tokenized_word=nltk.word_tokenize(text)
      stop_words=set(stopwords.words("french"))
      #Remove stop words
      filtered_sent=[]
      for w in tokenized_word:
          if w not in stop_words:
              filtered_sent.append(w)
      lemmatizer_word=[]
      lemmatizer=WordNetLemmatizer()
      for word in filtered_sent:
          lemmatizer_word.append(lemmatizer.lemmatize(word))
     
      leb=Label(root ,text=lemmatizer_word, font=("courrier",12)).pack(expand=YES) 
      root.mainloop()
 def stm():
      root = Tk()
      root.title("Stemming")
      root.geometry("1200x80")
      #root.minsize(480,360)
      root.maxsize(1200,80)
      root.iconbitmap("LG.ico")
      root.config()
      text="""Étant à l'arrêt au feu tricolore (rouge) j'ai été percuté à l'arrière par
         le véhicule B, son conducteur n'ayant pas réussi à s'arrêter avant mon véhicule.
         Mon véhicule a subi des dégâts visuels à l'arrière. Sous réserve d'autres dégâts."""
      # Removing puntuation
      translator_1 = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
      text= text.translate(translator_1)
      # Removing numbers
      text= re.sub(r'\d+', ' ',text)
      bad_chars = [' b ',' B ',' c ',' C ',' d ',' D ',' e ',' E ',' f ',' F ',' g ',' G ',' h ',' H ',' a ', ' A ',' i ',' I ',' j ',' J ',' K ',' k ',' l ',' L ',' m ',' M ',' n ',' N ',' o ',' O ',' p ',' P ',' q ',' Q ',' r ',' R ',' s ',' S ',' t ',' T ',' u ',' U ',' v ',' V ',' w ',' W ',' x ',' X ',' y ',' Y ',' z ',' Z' "*"] 
      for i in bad_chars : 
         text= re.sub(i , ' ',text)
      #print(text)
      tokenized_word=nltk.word_tokenize(text)
      stop_words=set(stopwords.words("french"))
      #Remove stop words
      filtered_sent=[]
      for w in tokenized_word:
          if w not in stop_words:
              filtered_sent.append(w)
      lemmatizer_word=[]
      lemmatizer=WordNetLemmatizer()
      for word in filtered_sent:
          lemmatizer_word.append(lemmatizer.lemmatize(word))
      ps = PorterStemmer()
      stemmed_words=[]
      for w in lemmatizer_word:
          stemmed_words.append(ps.stem(w)) 
      leb=Label(root ,text=stemmed_words, font=("courrier",12)).pack(expand=YES) 
      root.mainloop()
 def tf():
      text="""Étant à l'arrêt au feu tricolore (rouge) j'ai été percuté à l'arrière par
         le véhicule B, son conducteur n'ayant pas réussi à s'arrêter avant mon véhicule.
         Mon véhicule a subi des dégâts visuels à l'arrière. Sous réserve d'autres dégâts."""
     # Removing puntuation
      translator_1 = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
      text= text.translate(translator_1)
      # Removing numbers
      text= re.sub(r'\d+', ' ',text)
      bad_chars = [' b ',' B ',' c ',' C ',' d ',' D ',' e ',' E ',' f ',' F ',' g ',' G ',' h ',' H ',' a ', ' A ',' i ',' I ',' j ',' J ',' K ',' k ',' l ',' L ',' m ',' M ',' n ',' N ',' o ',' O ',' p ',' P ',' q ',' Q ',' r ',' R ',' s ',' S ',' t ',' T ',' u ',' U ',' v ',' V ',' w ',' W ',' x ',' X ',' y ',' Y ',' z ',' Z' "*"] 
      for i in bad_chars : 
          text= re.sub(i , ' ',text)
      #print(text)
      tokenized_word=nltk.word_tokenize(text)
      stop_words=set(stopwords.words("french"))
      #Remove stop words
      filtered_sent=[]
      for w in tokenized_word:
          if w not in stop_words:
              filtered_sent.append(w)
      lemmatizer_word=[]
      lemmatizer=WordNetLemmatizer()
      for word in filtered_sent:
          lemmatizer_word.append(lemmatizer.lemmatize(word))
      ps = PorterStemmer()
      stemmed_words=[]
      for w in lemmatizer_word:
          stemmed_words.append(ps.stem(w))
      fdist = FreqDist(stemmed_words)
      fdist.plot(30,cumulative=False)
      plt.show() 
 text = Button(window, text= " Le Texte ", font=("courrier",35),bg="#1DC6C8",fg="#34495E",command = txt)
 text.pack(expand=YES,fill=X)
 ph= Label(frame2, text= "      ", font=("courrier",15))
 ph.pack(expand=YES,fill=X)
 net=Button(frame2,text= "Nettoyage", font=("courrier",15),bg="#34495E" ,fg="white",command = net)
 net.pack(expand=YES,fill=X)
 ph= Label(frame2, text= "      ", font=("courrier",15))
 ph.pack(expand=YES,fill=X)
 tok=Button(frame2,text= "Tokenisation", font=("courrier",15),bg="#34495E" ,fg="white",command = tok)
 tok.pack(expand=YES,fill=X)
 ph= Label(frame2, text= "      ", font=("courrier",15))
 ph.pack(expand=YES,fill=X)
 net=Button(frame2,text= "Lemmatisation", font=("courrier",15),bg="#34495E" ,fg="white",command = lem)
 net.pack(expand=YES,fill=X)
 ph= Label(frame2, text= "      ", font=("courrier",15))
 ph.pack(expand=YES,fill=X)
 net=Button(frame2,text= "Stemming", font=("courrier",15),bg="#34495E" ,fg="white",command = stm)
 net.pack(expand=YES,fill=X)
 ph= Label(frame2, text= "      ", font=("courrier",15))
 ph.pack(expand=YES,fill=X)
 net=Button(frame2,text= "mesure TF", font=("courrier",15),bg="#34495E" ,fg="white",command = tf)
 net.pack(expand=YES,fill=X)

   
      



 canva=Canvas(frame1,width=600,height=500)
 image2=ImageTk.PhotoImage(Image.open("imm.png"))
 dic["image2"]=image2
 canva.create_image(10,10,anchor=NW,image=image2)
 canva.pack(expand=YES,fill=X)

 frame1.grid(row=0, column=0, sticky=W)
 frame2.grid(row=0, column=1, sticky=W)
 frame.pack(expand = YES)
  # affichage
 window.mainloop()


window1 =tk.Tk()
window1.title("Smart Constat")
window1.geometry("1350x800")
window1.minsize(1080,720)
window1.iconbitmap("LG.ico")
window1.config()
canvas=Canvas(window1)
image=ImageTk.PhotoImage(Image.open("ee.png"))
canvas.create_image(210,80,anchor=NW,image=image)
canvas.pack(expand=YES ,fill= BOTH)
label_tite2 = Label(canvas, text= "    ", font=("courrier",35))
label_tite2.grid(row=0,column=1, sticky=W)
label_tite2 = Label(canvas, text= "    ", font=("courrier",35))
label_tite2.grid(row=0,column=2, sticky=W)
label_tite2 = Label(canvas, text= "    ", font=("courrier",35))
label_tite2.grid(row=1,column=0, sticky=W)
label_tite2 = Label(canvas, text= "    ", font=("courrier",35))
label_tite2.grid(row=2,column=0, sticky=W)
label_tite2 = Label(canvas, text= "    ", font=("courrier",35))
label_tite2.grid(row=4,column=0, sticky=W)
label_titel = Label(canvas, text= "       Smart Constat", font=("courrier",40),fg="#2C3E50")
label_titel.grid(row=0,column=3, sticky=W)
button=Button(canvas,text= " Clustring ", font=("courrier",30),bg="#3498DB",fg="white",command=k_means)
button.grid(row=3,column=0, sticky=W)
but=Button(canvas,text= "Prétraitement", font=("courrier",23),bg="#E74C3C",fg="white",command=prét)
but.grid(row=5,column=0, sticky=W)

window1.mainloop()