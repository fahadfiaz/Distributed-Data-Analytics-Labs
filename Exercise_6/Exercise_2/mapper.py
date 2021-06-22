import sys
from string import punctuation

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
numbers='0123456789'
stop_words=['a','able','about','across','after','all','almost','also','am','among','an','and','any','are','as','at','be','because','been','but',
            'by','can','cannot','could','dear','did','do','does','either','else','ever','every','for','from','get','got','had','has','have','he',
            'her','hers','him','his','how','however','i','if','in','into','is','it','its','just','least','let','like','likely','may','me','might',
            'most','must','my','neither','no','nor','not','of','off','often','on','only','or','other','our','own','rather','said','say','says',
            'she','should','since','so','some','than','that','the','their','them','then','there','these','they','this','tis','to','too','twas',
            'us','wants','was','we','were','what','when','where','which','while','who','whom','why','will','with','would','yet','you','your']

no_punct_and_number = ""

# for line in sys.stdin:
    # line = line.strip()
    # for char in line:
        # if char not in punctuations and numbers:
            # no_punct_and_number = no_punct_and_number + char
    # words = no_punct_and_number.split()# split the line into words
    # for word in words:
        # if word not in stop_words:
            # word = (word.lower(), 1)
            # print(word)

for line in sys.stdin:
    line = line.translate(str.maketrans('','',punctuation))
    line = line.translate(str.maketrans('','','1234567890'))
    line = line.strip()# split the line into words
    words = line.split()
    for word in words:
        if word not in stop_words:
            word = (word.lower(), 1)
            print(word)
