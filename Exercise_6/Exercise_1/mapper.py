import sys

for line in sys.stdin:
    line = line.strip()
    words = line.split()  # split the line into words
    for word in words:
        word = (word.lower(), 1)
        print(word)
