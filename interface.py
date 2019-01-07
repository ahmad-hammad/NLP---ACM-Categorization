import urllib
import PyPDF2
import requests
from urllib.request import urlretrieve
from classify import train_all_models,predict_class
from word2vec import train_word2vec,predict_class_word2vec




import nltk 
#####################
def preprocess(line):

        #tokenizer 
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(line)

        words = [w.lower() for w in tokens]

        #stopwords
        stop_words = nltk.corpus.stopwords.words('english')
        filt = [w for w in words if not w in stop_words]

        # porter = nltk.PorterStemmer()
        # processesed1 = [porter.stem(t) for t in filt]

        wnl = nltk.WordNetLemmatizer()
        preprocessed = [wnl.lemmatize(t) for t in filt]

        return preprocessed


#############################
#retrieve document from acm library and show results 

def retrieve_acm(url):
    cookies = 0

    r = requests.get(url, cookies = cookies, headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.114 Safari/537.36'})

    f= open("doc.pdf","wb")
    f.write(r.content)


    f2 = open('D:\\Oulu\\NLP\\project\\workspace\\doc.pdf','rb')

    file_reader = PyPDF2.PdfFileReader(f2)

    text = ''
    #for i in range(file_reader.getNumPages()):
    for i in range(1):
        text = text + file_reader.getPage(i).extractText()

    return text

def show_results(text,classifiers):


 

    text = preprocess(text)
    preprocessed = ''
    for t in text:
        preprocessed =preprocessed+ " "+t
    preprocessed = [preprocessed]
    predict_class(preprocessed,classifiers,features)

def show_result_word2vec(text,classifiers_word2vec,model):  
    text = preprocess(text)
    preprocessed = ''
    for t in text:
        preprocessed =preprocessed+ " "+t
    preprocessed = [preprocessed]
    predict_class_word2vec(preprocessed,classifiers_word2vec,model)


#get user input
#https://dl.acm.org/ft_gateway.cfm?id=1298095&ftid=461125&dwn=1&CFID=10092248&CFTOKEN=cdee3755fc691456-5AEDDACC-0F06-2584-E8F69728D3802547 ##
print('start training models.......')
dataset_path = 'D:\\Oulu\\NLP\\project\\workspace\\out.txt'
#classifiers,features = train_all_models(dataset_path)

dataset_path_word2vec = 'D:\\Oulu\\NLP\\project\\workspace\\out_lemmas.txt'


classifiers_word2vec,model =  train_word2vec(dataset_path_word2vec)






url = input("please input the url of ACM pdf document\n")
file = retrieve_acm(url)

#show_results(file,classifiers)
show_result_word2vec(file,classifiers_word2vec,model)



#print(text)
url = input("please input another url or type exit\n")


while url!= 'exit':
    file = retrieve_acm(url)
    #show_results(file,classifiers)
    show_result_word2vec(file,classifiers_word2vec,model)
    url = input("please input another url or type exit\n")
