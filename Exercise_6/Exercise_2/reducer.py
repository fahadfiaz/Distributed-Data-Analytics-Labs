import sys
from ast import literal_eval
dic = {}
for line in sys.stdin:
   key = literal_eval(line)[0]
   dic[key] = dic.get(key , 0) + 1

for k,v in dic.items():
    print('{} {}'.format(k,v))