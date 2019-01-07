
import nltk 

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


def write(fw,out):
        for w in out:
                fw.write(w+" ")

f = open('D:\\Oulu\\NLP\\project\\dataset\\data.txt', encoding='utf-8')
fw = open('out_lemmas.txt','w+',encoding='utf-8')
data = [] 
labels = []


count =0
for line in f:


        if line != '':
                #count=count+1
                v=line.split('\t')
                for j in range(len(v)-1):
                        
                        out = preprocess(v[j])
                #         write(fw,out)
                #         fw.write('\t')
                # fw.write(v[len(v)-1])
                # #fw.write('\n')
        #print(count)
        # if count>100:
        print(line)
        print(out)
                break

                



