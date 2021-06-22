import sys
from ast import literal_eval


current_word = None
prev_filename = None
current_count = 0
N=0
file_name_with_total_wordCount={}
saved_previous_data=[]

for line in sys.stdin:
    saved_previous_data.append(line)
    filename,word,count=literal_eval(line)
    count=int(count)
    if prev_filename == filename:
        N=N+count
    else:
        if prev_filename != None:
            file_name_with_total_wordCount[prev_filename]=N
            #print(file_name_with_total_wordCount[prev_filename])
            
        N=0
        prev_filename = filename
    
    
file_name_with_total_wordCount[prev_filename]=N

for line in saved_previous_data:
    filename,word,count=literal_eval(line)
    for k in file_name_with_total_wordCount.keys():
        if filename == k:
           print((word,filename,count,file_name_with_total_wordCount[k]))
