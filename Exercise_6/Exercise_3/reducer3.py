import sys
from ast import literal_eval
from math import log10

prev_word = None
total_count = 1
word = None
df = {}
saved_previous_data = []

for line in sys.stdin:
    word,filename,wordCount,total_WordCount,count=literal_eval(line)
    if prev_word == word:
        total_count = total_count + int(count)
    else:
        if prev_word != None:
            df[prev_word] = (wordCount,total_WordCount,total_count)
            word_filename = (prev_word,filename) 
            saved_previous_data.append(word_filename)
        total_count = 1
        prev_word = word
        
df[prev_word] = (wordCount,total_WordCount,total_count)
word_filename = (prev_word,filename) 
saved_previous_data.append(word_filename)


for line in saved_previous_data:
    word,filename=line
    for k in df.keys():
        if word == k:
            #print((word,filename,df[k]))
            wordCount,total_WordCount,total_count_of_word_in_different_document=df[k]
            tfidf= (wordCount/total_WordCount)*log10(5/total_count_of_word_in_different_document)
            print('{}: {}'.format(word,tfidf))
