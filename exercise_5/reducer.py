import sys
from ast import literal_eval
from collections import Counter 

airport_Names = []
final = {}
for line in sys.stdin:
    line = literal_eval(line) # convert string-tuple to tuple 
    try:
        line = (line[0], float(line[1]),float(line[2]))
    except Exception as e:
        line=None
    if line:
        if line[0] in final.keys():
            old_val = final[line[0]]
            airport =old_val[0]
            low=old_val[1]
            high=old_val[2]
            total=old_val[3]+line[1]
            count=old_val[4]+1
            total_arrival_delay=old_val[5]+line[2]
            if line[1] < old_val[1]:
                low=line[1]
            if line[1] > old_val[2]:
                high = line[1]
            
            final[line[0]]=(airport,low,high,total,count,total_arrival_delay)

        else:
            # airport_Names.append(line[0])
            final[line[0]] = (line[0],9999,-9999,0,1,0)

#show final uotput
# for k,v in final.items():
    # print("Airport Name:{} ,maximum departure delay:{} ,maximum departure delay:{}, averaga departure delay:{}, averaga arrival delay:{}".format(v[0],v[1],v[2],v[3]/v[4],v[5]/v[4]))


#calculate top 10 airports by their average Arrival delay.

Arrival_Delay_Average={}
for k,v in final.items():
    Arrival_Delay_Average[v[0]]=v[5]/v[4]

d = Counter(Arrival_Delay_Average)
for k, v in d.most_common(10):
    print("Airport Name:{}, averaga arrival delay:{}".format(k,v))


    
