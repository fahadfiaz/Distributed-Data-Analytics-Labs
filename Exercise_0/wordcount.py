from collections import Counter
import matplotlib.pyplot as plt


def my_word_count(words):
    counter = {}
    for word in words:
        if not word in counter:
            counter[word] = 1
        else:
            counter[word] += 1

    print(counter)


file = open("word_count.txt", "r")  # Open file in read mode
totalWords = file.read().split()  # Read data from file and split it into words. Here we have not passed any argument to split function. In this case split function will consider consecutive whitespace as a single separator, and the result will contain no empty strings at the start or end if the string has leading or trailing whitespace. Consequently, splitting an empty string or a string consisting of just whitespace with a None separator returns [].
stopwords = ['the', 'a', 'an', 'be']  # list of words to exclude from totalWords
reduced = [words for words in totalWords
           if words.lower() not in stopwords]

Top10wordcount = dict(
    Counter(reduced).most_common(
        10))  # dictionary subclass Counter used for counting occurence of words. Then applying most_common function which return list of the 10 most common elements and their counts
plt.style.use('fivethirtyeight')  # using pre-defined style provided by Matplotlib.
plt.title("Occurances of Unique Words")  # Set title for the axes.
plt.xlabel("Words")  # Set the label for the x-axis.
plt.ylabel("Count")  # Set the label for the y-axis.
plt.tight_layout(0)  # used to give padding
plt.bar(Top10wordcount.keys(), Top10wordcount.values(),
        color='#30475e', edgecolor="black")  # bar graph to plot chart
plt.show()  # show chart
