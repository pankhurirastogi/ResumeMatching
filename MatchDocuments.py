from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

def processFiles(document):
	raw = open(document).read()
	tokens = word_tokenize(raw)
	words = [w.lower() for w in tokens]

	porter = nltk.PorterStemmer()
	stemmed_tokens = [porter.stem(t) for t in words]

	stop_words = set(stopwords.words('english'))
	filtered_tokens = [w for w in stemmed_tokens if not w in stop_words]

	count = nltk.defaultdict(int)
	for word in filtered_tokens:
		count[word]+=1
	return count



def cos_sim(a,b):
	dot_product = np.dot(a,b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product/(norm_a*norm_b)



def findSimilarity(dict1, dict2):
	all_words_list = []
	for key in dict1:
		all_words_list.append(key)
	for key in dict2:
		all_words_list.append(key)

	all_words_list_size = len(all_words_list)
	vector1 = np.zeros(all_words_list_size, dtype=np.int)
	vector2 = np.zeros(all_words_list_size, dtype=np.int)

	i=0
	for (key) in all_words_list:
		vector1[i]=dict1.get(key,0)
		vector2[i] = dict2.get(key,0)

	return cos_sim(vector1,vector2)


dict1 = processFiles("a1.txt")
dict2 = processFiles("b1.txt")
print findSimilarity(dict1, dict2)




