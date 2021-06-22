import sys
from ast import literal_eval

for line in sys.stdin:                       
    filename_word_count = literal_eval(line)
    print(filename_word_count)