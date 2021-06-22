import sys
from ast import literal_eval

for line in sys.stdin:                       
    word_filename_count_totalcount = literal_eval(line)
    print((word_filename_count_totalcount[0],word_filename_count_totalcount[1],word_filename_count_totalcount[2],word_filename_count_totalcount[3],1))