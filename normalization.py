import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize

stopWords = set(stopwords.words('portuguese'))

def pontuacion(word):
    tokenizer = RegexpTokenizer(r'\w+')
    word = tokenizer.tokenize(word)
    return " ".join(word)

def remove_link(l):
    join = []
    new = []
    for i in l:
        temp = i.split()
        for j in temp:
            j = re.sub(r'^https?:\/\/.*[\r\n]*', '', j, flags=re.MULTILINE)
            j = re.sub(r'\@.*[\r\n]*', '', j)
            j = "".join(j)
            join.append(j)
        new.append(" ".join(join))
        join = []
    return new

def normalize(l):
    l = remove_link(l)
    new = []
    for data in l:
        data = pontuacion(data)
        words = word_tokenize(data, language='portuguese')
        wordsFiltered = []
        for w in words:
            if w not in stopWords:
                wordsFiltered.append(w)
        new.append(" ".join(wordsFiltered))

    return new


# l = ["Glória ao soberano a Senhor. https://t.co/N9zbtGLuWs @lovekmalskdflk", "amor, a de"]

# d = 'Eighty-seven miles to go, yet.  Onward!'
# print(pontuacion(d))
# normalize(l)