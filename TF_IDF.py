from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, linear_model, naive_bayes, metrics, svm, preprocessing, ensemble, feature_selection
import pandas, numpy
import statistics 


#f = open('C:\\Users\\Abdelrahman\\Downloads\\Documents\\NLP\\Project\\ACM_multilabel_dataset_2008-utf8.tsv', encoding='utf-8')

    #utility function for model training
def train_model(classifier, feature_vector_train, label, feature_vector_test,test_y, is_neural_net=False):

        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)

        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_test)

        if is_neural_net:
            predictions = predictions.argmax(axis=-1)

        return [metrics.accuracy_score(test_y, predictions),metrics.precision_score(test_y, predictions, average='macro'),metrics.recall_score(test_y, predictions, average='macro'),metrics.f1_score(test_y, predictions, average='macro'),metrics.hamming_loss(test_y, predictions)], classifier

 

def train_all_models(dataset_path):
    classifiers = []
    features = [] 
    f = open(dataset_path, encoding='utf-8')
    data = [] 
    labels = []
    count = 0
    for line in f:
        if line != '':
            #v = f.readline()
            v=line.split('\t')
            #if len(v)<5: line
            if count >= 1000:
                break
            if v[3] != '': 
                data.append(" ".join(v[:3]))
                labels.append(v[3])
                count = count + 1
            
        

    #print(labels)

    # create a dataframe using data and lables
    trainDF = pandas.DataFrame()
    trainDF['text'] = data
    trainDF['label'] = labels
    accuracy_Bayes_Count,accuracy_Bayes_IDF,accuracy_Bayes_NGram,accuracy_SVM_Count,accuracy_SVM_IDF,accuracy_SVM_NGram,accuracy_LR_Count,accuracy_LR_IDF,accuracy_LR_NGram,accuracy_RF_Count,accuracy_RF_IDF,accuracy_RF_NGram = ([] for i in range(12))
    numFeatures = 500
    maxFeatures = numFeatures


    train_x, test_x, train_y, test_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size = 0.10)



        # create a count vectorizer object 
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=maxFeatures)
    count_vect.fit(trainDF['text'])

    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    # print(xtrain_count.toarray() )
    xtest_count =  count_vect.transform(test_x)


    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=maxFeatures)  #maybe needs to be changed
    tfidf_vect.fit(trainDF['text'])
   # print(tfidf_vect.vocabulary_ )
    
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    #print(type(xtrain_tfidf))
    xtest_tfidf =  tfidf_vect.transform(test_x)
   # selector_tfidf = feature_selection.SelectKBest(feature_selection.mutual_info_classif, k=numFeatures)
   # xtrain_tfidf = selector_tfidf.fit_transform(xtrain_tfidf, train_y)
   # print(xtrain_tfidf.shape)
    #xtest_tfidf = selector_tfidf.transform(xtest_tfidf)
   # print(xtest_tfidf.shape)
    
    # ngram level tf-idf 
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=maxFeatures) #maybe needs to be changed
    tfidf_vect_ngram.fit(trainDF['text'])
    #print(tfidf_vect_ngram.vocabulary_ )
    
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
    xtest_tfidf_ngram =  tfidf_vect_ngram.transform(test_x)
    #selector_tfidf_ngram = feature_selection.SelectKBest(feature_selection.mutual_info_classif, k=numFeatures)
    #xtrain_tfidf_ngram = selector_tfidf_ngram.fit_transform(xtrain_tfidf_ngram, train_y)
    # print(xtrain_tfidf.shape)
    #xtest_tfidf_ngram = selector_tfidf_ngram.transform(xtest_tfidf_ngram)
    # # load the pre-trained word-embedding vectors 
    # embeddings_index = {}
    # for i, line in enumerate(open('data/wiki-news-300d-1M.vec', encoding="utf8")):
    # values = line.split()
    # embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

    # # create a tokenizer 
    # token = text.Tokenizer()
    # token.fit_on_texts(trainDF['text'])
    # word_index = token.word_index
    # print(len(word_index))
    # print(word_index)
    # # convert text to sequence of tokens and pad them to ensure equal length vectors 
    # train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=5000)
    # test_seq_x = sequence.pad_sequences(token.texts_to_sequences(test_x), maxlen=5000)
    # print(len(test_seq_x))
    # # create token-embedding mapping
    # embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
    # for word, i in word_index.items():
    # embedding_vector = embeddings_index.get(word)
    # if embedding_vector is not None:
        # embedding_matrix[i] = embedding_vector

    features = [count_vect,tfidf_vect,tfidf_vect_ngram] 



    # Naive Bayes on Count Vectors
    accuracy_Bayes_Count,classifier_Bayes_Count = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xtest_count,test_y)
    print ("NB, Count Vectors: ", accuracy_Bayes_Count)
    classifiers.append(classifier_Bayes_Count)

    # Naive Bayes on Word Level TF IDF Vectors
    accuracy_Bayes_IDF,classifier_Bayes_IDF = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xtest_tfidf,test_y)
    print("NB, WordLevel TF-IDF: ", accuracy_Bayes_IDF)
    classifiers.append(classifier_Bayes_IDF)

    # Naive Bayes on Ngram Level TF IDF Vectors
    accuracy_Bayes_NGram,classifier_Bayes_NGram = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram,test_y)
    print ("NB, N-Gram Vectors: ", accuracy_Bayes_NGram)
    classifiers.append(classifier_Bayes_NGram)

    # SVM on Count Vectors
    accuracy_SVM_Count,classifier_SVM_Count =  train_model(svm.SVC(), xtrain_count, train_y, xtest_count,test_y)
    print ("SVM, Count Vectors: ", accuracy_SVM_Count)
    classifiers.append(classifier_SVM_Count)

    # SVM on Word Level TF IDF Vectors
    accuracy_SVM_IDF,classifier_SVM_IDF  = train_model(svm.SVC(), xtrain_tfidf, train_y, xtest_tfidf,test_y)
    print ("SVM, WordLevel TF-IDF: ", accuracy_SVM_IDF)
    classifiers.append(classifier_SVM_IDF)

    # SVM on Ngram Level TF IDF Vectors
    accuracy_SVM_NGram,classifier_SVM_NGram =train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram,test_y)
    print ("SVM, N-Gram Vectors: ", accuracy_SVM_NGram)
    classifiers.append(classifier_SVM_NGram)

    # Linear Classifier on Count Vectors
    accuracy_LR_Count,classifier_LR_Count =train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xtest_count,test_y)
    print ("LR, Count Vectors: ", accuracy_LR_Count)
    classifiers.append(classifier_LR_Count)

    # Linear Classifier on Word Level TF IDF Vectors
    accuracy_LR_IDF,classifier_LR_IDF = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xtest_tfidf,test_y)
    print ("LR, WordLevel TF-IDF: ", accuracy_LR_IDF)
    classifiers.append(classifier_LR_IDF)

    # Linear Classifier on Ngram Level TF IDF Vectors
    accuracy_LR_NGram,classifier_LR_NGram =train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram,test_y)
    print ("LR, N-Gram Vectors: ", accuracy_LR_NGram)
    classifiers.append(classifier_LR_NGram)

    # RF on Count Vectors
    accuracy_RF_Count,classifier_RF_Count =train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xtest_count,test_y)
    print ("RF, Count Vectors: ", accuracy_RF_Count)
    classifiers.append(classifier_RF_Count)

    # RF on Word Level TF IDF Vectors
    accuracy_RF_IDF,classifier_RF_IDF =train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xtest_tfidf,test_y)
    print ("RF, WordLevel TF-IDF: ", accuracy_RF_IDF)
    classifiers.append(classifier_RF_IDF)

    # RF on Ngram Level TF IDF Vectors
    accuracy_RF_NGram,classifier_RF_NGram =train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram,test_y)
    print ("RF, N-Gram Vectors: ", accuracy_LR_NGram)
    classifiers.append(classifier_RF_NGram)

    return classifiers,features




def predict_class(doc,classifiers,features):

    
    #count based
    xsample_count =  features[0].transform(doc)
    
    # word level tf-idf   
    xsample_tfidf =  features[1].transform(doc)

    # ngram level tf-idf
    xsample_tfidf_ngram =  features[2].transform(doc)



    # Naive Bayes on Count Vectors
    print ("NB, Count Vectors: ", classifiers[0].predict(xsample_count))
    

    # Naive Bayes on Word Level TF IDF Vectors
    print("NB, WordLevel TF-IDF: ", classifiers[1].predict(xsample_tfidf))


    # Naive Bayes on Ngram Level TF IDF Vectors
    print ("NB, N-Gram Vectors: ", classifiers[2].predict(xsample_tfidf_ngram))

    # SVM on Count Vectors
    print ("SVM, Count Vectors: ", classifiers[3].predict(xsample_count))


    # SVM on Word Level TF IDF Vectors
    print ("SVM, WordLevel TF-IDF: ", classifiers[4].predict(xsample_tfidf))

    # SVM on Ngram Level TF IDF Vectors
    print ("SVM, N-Gram Vectors: ", classifiers[5].predict(xsample_tfidf_ngram))

    # Linear Classifier on Count Vectors
    print ("LR, Count Vectors: ", classifiers[6].predict(xsample_count))

    # Linear Classifier on Word Level TF IDF Vectors
    print ("LR, WordLevel TF-IDF: ", classifiers[7].predict(xsample_tfidf))
    
    # Linear Classifier on Ngram Level TF IDF Vectors
    print ("LR, N-Gram Vectors: ", classifiers[8].predict(xsample_tfidf_ngram))

    # RF on Count Vectors
    print ("RF, Count Vectors: ", classifiers[9].predict(xsample_count))

    # RF on Word Level TF IDF Vectors
    print ("RF, WordLevel TF-IDF: ", classifiers[10].predict(xsample_tfidf))

    # RF on Ngram Level TF IDF Vectors
    print ("RF, N-Gram Vectors: ", classifiers[11].predict(xsample_tfidf_ngram))


