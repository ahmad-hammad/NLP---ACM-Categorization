import numpy
import math
from gensim.models import Word2Vec
# embeddings_index = {}
# for i, line in enumerate(open('E:/GoogleNews-vectors-negative300.bin','rb')):
#     values = line.split()
#     embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')
from gensim.models.keyedvectors import KeyedVectors

#model = Word2Vec.load_word2vec_format('E:/GoogleNews-vectors-negative300.bin', binary=True)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, linear_model, naive_bayes, metrics, svm, preprocessing, ensemble
import pandas
import scipy.sparse


def get_doc_vec(terms,model,Print_f = False):
    not_found=0
    count=1
    doc_vect= numpy.zeros(300)
    tokens = terms.split()
    for term in tokens:
        try:
            
            doc_vect = doc_vect + model[term]
            count+=1
        except:
            not_found+=1
            pass
    doc_vect=doc_vect/count
    if Print_f:
        print('not found count ',not_found)
    return doc_vect

#utility function for model training
def train_model(classifier, feature_vector_train, label, feature_vector_valid,test_y, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return [metrics.accuracy_score(test_y, predictions),metrics.precision_score(test_y, predictions, average='macro'),metrics.recall_score(test_y, predictions, average='macro'),metrics.f1_score(test_y, predictions, average='macro'),metrics.hamming_loss(test_y, predictions)],classifier
    

size  = 1000
def train_word2vec(dataset_path):

    f = open(dataset_path, encoding='utf-8')
    data = [] 
    labels = []
    count = 0
    for line in f:
        if line != '':
            #v = f.readline()
            v=line.split('\t')
            #if len(v)<5: line
            if count >= size:
                break
            if v[3] != '': 
                data.append(" ".join(v[0:3]))
                labels.append(v[3])
                count = count + 1
            
            



    # create a dataframe using data and lables
    trainDF = pandas.DataFrame()
    trainDF['text'] = data
    trainDF['label'] = labels

    # split the dataset into training and validation datasets 
    train_x, test_x, train_y, test_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size = 0.20)

    #print(len(test_y))





    # # Naive Bayes on Count Vectors
    # accuracy_Bayes_Count = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xtest_count)
    # print ("NB, Count Vectors: ", accuracy_Bayes_Count)

    # Naive Bayes on Word Level TF IDF Vectors
    #accuracy_Bayes_IDF = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xtest_tfidf)
    model = KeyedVectors.load_word2vec_format('E:/GoogleNews-vectors-negative300.bin', binary=True,limit=200)
    ratio =.8 
    doc_matrix = numpy.zeros((math.ceil(ratio*size),300))
    doc_matrix_test = numpy.zeros((math.ceil((1-ratio)*size),300))
    i=0
    for t in train_x:
        doc_vec=get_doc_vec(t,model)
        doc_matrix[i]=doc_vec
        i+=1


    # doc_vec=get_doc_vec(train_x.get(1),model)
    # doc_matrix[0] = doc_vec
    #print(doc_matrix[1])
    train_vec = scipy.sparse.csr.csr_matrix(doc_matrix)
    #print(model['home'].shape)

    i=0
    for t in test_x:
        doc_vec=get_doc_vec(t,model)
        doc_matrix_test[i]=doc_vec
        i+=1

    test_vec = scipy.sparse.csr.csr_matrix(doc_matrix_test)

    print(type(test_vec))
    # home = model['home']
    # print(home)
    # print(len(home))
    # print('sucess')
    classifiers= []

    # accuracy_Bayes_IDF = train_model(linear_model.LogisticRegression(),train_vec , train_y, test_vec)
    # print("NB: ", accuracy_Bayes_IDF)

    # Naive Bayes 
    accuracy_Bayes_Count,classifier_Bayes_Count = train_model(naive_bayes.MultinomialNB(), abs(train_vec), train_y, abs(test_vec),test_y)
    print ("NB: ", accuracy_Bayes_Count)
    classifiers.append(classifier_Bayes_Count)



    # SVM 
    accuracy_SVM_Count,classifier_SVM_Count =  train_model(svm.SVC(), abs(train_vec), train_y, abs(test_vec),test_y)
    print ("SVM: ", accuracy_SVM_Count)
    classifiers.append(classifier_SVM_Count)



    # Linear Classifier 
    accuracy_LR_Count,classifier_LR_Count =train_model(linear_model.LogisticRegression(), train_vec, train_y, test_vec,test_y)
    print ("LR: ", accuracy_LR_Count)
    classifiers.append(classifier_LR_Count)



    # RF 
    accuracy_RF_Count,classifier_RF_Count =train_model(ensemble.RandomForestClassifier(), train_vec, train_y, test_vec,test_y)
    print ("RF: ", accuracy_RF_Count)
    classifiers.append(classifier_RF_Count)

    return classifiers,model


def predict_class_word2vec(doc, classifiers,model):
    doc_vec=get_doc_vec(doc[0],model,True)

    doc_vec_transformed = scipy.sparse.csr.csr_matrix(doc_vec)
    print("Results of using word embedding as feature set")
        
    # Naive Bayes 
    accuracy_Bayes = classifiers[0].predict(doc_vec_transformed)
    print ("NB: ", accuracy_Bayes)



    # SVM 
    accuracy_SVM = classifiers[1].predict(doc_vec_transformed)
    print ("SVM: ", accuracy_SVM)



    # Linear Classifier 
    accuracy_LR= classifiers[2].predict(doc_vec_transformed)
    print ("LR: ", accuracy_LR)



    # RF 
    accuracy_RF = classifiers[3].predict(doc_vec_transformed)
    print ("RF: ", accuracy_RF)
