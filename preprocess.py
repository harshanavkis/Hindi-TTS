import codecs
import numpy as np #include dict for ,

def wordToCode(sentence):
	sentence = sentence.split(" ")
	sentCode = []
	for word in sentence:
		wordCode = [ord(i) for i in word]
		print(wordCode)
		sentCode.append(wordCode)
	return np.asarray(sentCode)

def getChars(fileName):
	text = codecs.open(fileName, encoding = "utf-8").read()
	text = text.split("\n")
	text = [i.split(' \" ') for i in text]
	text = [i[1] for i in text if len(i) > 1]
	text = [wordToCode(i) for i in text]
	return np.asarray(text)