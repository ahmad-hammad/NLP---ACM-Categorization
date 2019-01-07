
from nltk.metrics import *
from thesaurus import Word
import nltk 

f = open('D:\\Oulu\\NLP\\project\\workspace\\out_lemmas.txt', encoding='latin2')
data = [] 

count = 0 
for line in f:
    count+=1
    if line != '' :
        #v = f.readline()
        v=line.split('\t')
        #if len(v)<5: line
        if v[3] != '' : 
            data.append(v)
    if count>=5000:
        break



print('document number: ' ,len(data))

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

classes =[] 
f2 = open("D:\\Oulu\\NLP\\project\\classes.txt") 
for line in f2:
    classes.append(line)

sims =  []
# for i in range(len(data)-1):
#     for j in range(len(classes)):
#         s1 = set(data[i+1][1].split())
#         s2 = set(classes[j].split()[1:])
#         sim = 1 -jaccard_distance(s1, s2)
#         sims.append(sim)

print('title without synonyms applied')
correct_predictions = 0
for i in range(len(data)):
    actual_class = data[i][3]
    predicted_class = []
    current_sim = 0
    for j in range(len(classes)):
        
        s1 = set(data[i][0].split())
        s2 = set(classes[j].split()[1:])
        
        sim = 1 -jaccard_distance(s1, s2)
        if sim!=0:
            if sim==current_sim:
                predicted_class+=classes[j]
            if sim > current_sim:
                current_sim =sim
                predicted_class=classes[j]
    # if len(predicted_class)>0 and i==0:
    #     print(predicted_class,i+1)

    if actual_class in predicted_class:
        correct_predictions+=1
print(correct_predictions)

def  get_syn(words):
    syns =[] 
    for i in range(len(words)):
        try: 
            w =  Word(words[i])
            syn=  w.synonyms('all')
            for j in range(len(syn)):
                syns =syns+syn[j]
        except:
            #print(words[i],' not found')
            pass
    words = words+syns
    return words

def  get_syn_classes(words):
    syns =[] 
    for i in range(len(words)):
        try: 
            w =  Word(words[i])
            syn=  w.synonyms('all')
            for j in range(len(syn)):
                syns =syns+syn[j]
        except:
            pass
    words = words+syns
    return words
 
 
            



classes_syn = [] 
classes_splitted = []
# for j in range(len(classes)):
#     tokens = tokenizer.tokenize(classes[j])
#     words = [w.lower() for w in tokens]
#     classes_syn.append(get_syn( words[2:]))



f3 = open("D:\\Oulu\\NLP\\project\\workspace\\out_classSyn.txt") 
for line in f3:
    classes_syn.append(line.split())
f3.close

print("finished")
for j in range(len(classes)):
    classes_splitted.append(set(classes[j].split()[1:]))

print('title')
correct_predictions = 0
for i in range(len(data)):
    actual_class = data[i][3]
    predicted_class = []
    current_sim = 0
    for j in range(len(classes)):
        
        s1 = set(data[i][0].split())
        s2 = classes_splitted[j]
        syns = set(classes_syn[j])
        num_int = len(s1.intersection(syns))
        
        sim = num_int/len(s1.union(s2))
        if sim!=0:
            if sim==current_sim:
                predicted_class+=classes[j]
            if sim > current_sim:
                current_sim =sim
                predicted_class=classes[j]
    # if len(predicted_class)>0 and i==0:
    #     print(predicted_class,i+1)

    if actual_class in predicted_class:
        correct_predictions+=1
print(correct_predictions)

print('title+abstract')
#title+abstract
correct_predictions = 0
for i in range(len(data)):
    actual_class = data[i][3]
    predicted_class = []
    current_sim = 0
    for j in range(len(classes)):
        
        s1 = set(data[i][0].split()+data[i][1].split())
        s2 = classes_splitted[j]
        syns = set(classes_syn[j])
        num_int = len(s1.intersection(syns))
        
        sim = num_int/len(s1.union(s2))
        if sim!=0:
            if sim==current_sim:
                predicted_class+=classes[j]
            if sim > current_sim:
                current_sim =sim
                predicted_class=classes[j]
    # if len(predicted_class)>0 and i==0:
    #     print(predicted_class,i+1)

    if actual_class in predicted_class:
        correct_predictions+=1
print(correct_predictions)

print('title + abstract + keywords')
#title + abstract + keywords
correct_predictions = 0
for i in range(len(data)):
    actual_class = data[i][3]
    predicted_class = []
    current_sim = 0
    for j in range(len(classes)):
        
        s1 = set(data[i][0].split()+data[i][1].split()+data[i][2].split())
        s2 = classes_splitted[j]
        syns = set(classes_syn[j])
        num_int = len(s1.intersection(syns))
        
        sim = num_int/len(s1.union(s2))
        if sim!=0:
            if sim==current_sim:
                predicted_class+=classes[j]
            if sim > current_sim:
                current_sim =sim
                predicted_class=classes[j]
    # if len(predicted_class)>0 and i==0:
    #     print(predicted_class,i+1)

    if actual_class in predicted_class:
        correct_predictions+=1
print(correct_predictions)

print('keywords')
#keywords
correct_predictions = 0
for i in range(len(data)):
    actual_class = data[i][3]
    predicted_class = []
    current_sim = 0
    for j in range(len(classes)):
        
        s1 = set(data[i][2].split())
        s2 = classes_splitted[j]
        syns = set(classes_syn[j])
        num_int = len(s1.intersection(syns))
        
        sim = num_int/len(s1.union(s2))
        if sim!=0:
            if sim==current_sim:
                predicted_class+=classes[j]
            if sim > current_sim:
                current_sim =sim
                predicted_class=classes[j]
    # if len(predicted_class)>0 and i==0:
    #     print(predicted_class,i+1)

    if actual_class in predicted_class:
        correct_predictions+=1
print(correct_predictions)

print('abstract')
#abstract
correct_predictions = 0
for i in range(len(data)):
    actual_class = data[i][3]
    predicted_class = []
    current_sim = 0
    for j in range(len(classes)):
        
        s1 = set(data[i][1].split())
        s2 = classes_splitted[j]
        syns = set(classes_syn[j])
        num_int = len(s1.intersection(syns))
        
        sim = num_int/len(s1.union(s2))
        if sim!=0:
            if sim==current_sim:
                predicted_class+=classes[j]
            if sim > current_sim:
                current_sim =sim
                predicted_class=classes[j]
    # if len(predicted_class)>0 and i==0:
    #     print(predicted_class,i+1)

    if actual_class in predicted_class:
        correct_predictions+=1
print(correct_predictions)


#print (get_syn(['good','boy']))

# #title with synonms

# def check_siml(word1,word2):
#     w =  Word(word2)
#     syns=  w.synonyms('all')

#     for i in range(len(syns)):
#         for j in range(len(syns[i])):
#             if word1 == syns[i][j]:
#                 return True
    
#     return False


# ch = check_siml('bad','right')
# print(ch)

#keywords+abstaract +keyword




def write(fw,out):
        for w in out:
                fw.write(w+" ")



# fw = open('out_classSyn.txt','w+')

# for i  in range(len(classes_syn)):
#     write(fw,classes_syn[i])
#     fw.write('\n')



