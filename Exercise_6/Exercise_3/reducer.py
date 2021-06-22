import sys
from ast import literal_eval
dic = {}
for line in sys.stdin:
   key = literal_eval(line)[:2]
   dic[key] = dic.get(key , 0) + 1

for k,v in dic.items():
    print((k[0],k[1],v))