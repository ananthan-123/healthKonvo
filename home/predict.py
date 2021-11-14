from ast import literal_eval
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import pickle
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
df = {}


# def editSymptom(text):
#     #     print(type(text))
#     if type(text) == str:
#         text = nltk.word_tokenize(text)
#         pos_tagged = nltk.pos_tag(text)
# #         print(pos_tagged)
#         nouns = filter(lambda x: x[1] == 'NN' or x[1] == 'NNS' or x[1] == 'NNP' or x[1] == 'NNPS' or x[1] == 'PRP' or x[1] == 'PRP$' or x[1] == 'RB' or x[1]
#                        == 'RBR' or x[1] == 'RBS' or x[1] == 'VB' or x[1] == 'VBG' or x[1] == 'VBD' or x[1] == 'VBN' or x[1] == 'VBP' or x[1] == 'VBZ', pos_tagged)
# #         print(list(nouns))
#         result = []
#         for i in nouns:
#             result.append(i[0])
#         return result
#     return None


def editSymptom(text):
    #     print(text)
    if type(text) == str:
        text = text.lower()
        text = nltk.word_tokenize(text)

#         print(text)

        res = []

        for i in text:
            b = TextBlob(i)
            res.append(str(b.correct()))

        lemmatizer = WordNetLemmatizer()
        result = []
        result1 = []
        unwanted = ['i', 'am', 'be', 'are', 'is', 'was', 'were', 'being', 'can', 'could', 'do', 'did', 'does', 'doing',
                    'have', 'had', 'has', 'having', 'may', 'might', 'must', 'shall', 'should', 'will', 'would', 'days', 'day']
        pos_tagged = nltk.pos_tag(res)

        text = filter(lambda x: x[1] == 'NN' or x[1] == 'NNS' or x[1] == 'NNP' or x[1] == 'JJ' or x[1] == 'NNPS' or x[1] == 'PRP' or x[1] == 'PRP$' or x[1] ==
                      'RB' or x[1] == 'RBR' or x[1] == 'RBS' or x[1] == 'VB' or x[1] == 'VBG' or x[1] == 'VBD' or x[1] == 'VBN' or x[1] == 'VBP' or x[1] == 'VBZ', pos_tagged)
        for i in text:
            if i[0] not in unwanted:
                tem = i[0].replace('.', '')
                result1.append(tem)
        return result1

    else:
        return []


def main():
    global df
    df = pd.read_csv(r'home/final-disease.csv',
                     converters={"symptoms": literal_eval})
    print('done')


def predictOut(text):
    edited = editSymptom(text)
    # print(edited)
    edited = ' '.join(edited)
    res = process.extract(edited, df['symptoms'], limit=10)

    ind = res[0][2]

    dis = df.iloc[ind]

    print(dis[0])

    resp = 'I think you have '+dis[0]

    return resp


main()
