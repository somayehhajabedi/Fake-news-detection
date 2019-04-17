import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import neural_network
from sklearn.neighbors import KNeighborsClassifier
from gensim import utils
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import LabeledSentence


class FakeNewsDetector:

    def __init__(self, trainFile):
        self.stopwords = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've",
                              "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he", "him", "his",
                              "himself",
                              "she", "she's", "her", "hers", "herself", "it", "it's", "its", "itself", "they", "them",
                              "their",
                              "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "that'll",
                              "these", "those",
                              "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
                              "do", "does",
                              "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
                              "while", "of",
                              "at", "by", "for", "with", "about", "against", "between", "into", "through", "during",
                              "before",
                              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
                              "under",
                              "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
                              "any",
                              "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
                              "only", "own",
                              "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't",
                              "should",
                              "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't",
                              "couldn",
                              "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't",
                              "haven",
                              "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn",
                              "needn't",
                              "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won",
                              "won't",
                              "wouldn", "wouldn't"])
        self.trainFile = trainFile

    def extractFeatures(self):
        self.df = pd.read_csv(self.trainFile, header=0)
        # print(self.df.count())
        self.df = self.df.head(5000)
        missing_rows = []
        for item in range(len(self.df)):
            if self.df.loc[item, "text"] != self.df.loc[item, "text"]:
                missing_rows.append(item)
        self.df = self.df.drop(missing_rows).reset_index().drop(["index", "id"], axis=1)

        self.doc2vecModel = Doc2Vec(min_count=1, window=5, vector_size=300, sample=1e-4, negative=5, workers=7,
                                    epochs=10, seed=1)

        self.sentenceList = []
        for index, row in self.df["text"].iteritems():
            row_split = utils.to_unicode(row).split()
            self.sentenceList.append(LabeledSentence(row_split, ["Text" + "_%s" % str(index)]))

        # print( self.sentenceList)

        self.doc2vecModel.build_vocab(self.sentenceList)
        self.doc2vecModel.train(self.sentenceList, total_examples=self.doc2vecModel.corpus_count,
                                epochs=self.doc2vecModel.iter)

        self.featureList = []
        for i in range(len(self.sentenceList)):
            self.featureList.append(self.doc2vecModel.docvecs["Text_" + str(i)])
        # print(self.featureList,self.df["label"])
        # print('================>>')
        # print(len(self.featureList))
        return self.featureList, self.df["label"]

    def train(self, type, h,h2):
        featureList, labelList = self.extractFeatures()
        if type == "NeuralNetwork":
            self.classifier = neural_network.MLPClassifier(hidden_layer_sizes=(10,10)
                                                           # , activation="relu",
                                                           # learning_rate="consta1nt",learning_rate_init=0.001,
                                                           ,max_iter=10
                                                           )
        elif type == "KNN":
            self.classifier = KNeighborsClassifier(n_neighbors=h)

        self.classifier.fit(featureList, labelList)

        scores = cross_val_score(self.classifier, featureList, labelList, cv=5)
        print( "type:%s %s , %s   Mean:%0.4f and std:%0.2f  " % (type,h,h2,scores.mean(), scores.std() * 2))


    def predict(self, news, type, h,h2):
        self.train(type, h,h2)
        model = self.doc2vecModel.infer_vector(list(news)).reshape(1, -1)
        predict = self.classifier.predict(model)
        print(predict)


trainFile = "D:\\bigData\\assignment\\bigdata-la3-w2019-minaboori\\data\\train.csv"

clf = FakeNewsDetector(trainFile)

# clf.train("NeuralNetwork",0)
clf.predict("asdfsdfsdgsdgsdg", "NeuralNetwork", 50,0)
# clf.predict("asdfsdfsdgsdgsdg", "NeuralNetwork", 1000,1000)
# clf.predict("asdfsdfsdgsdgsdg", "NeuralNetwork", 3000,3000)
# # clf.train("KNN",7)
# clf.predict("asdfsdfsdgsdgsdg", "KNN", 2,0)
# clf.predict("asdfsdfsdgsdgsdg", "KNN", 3,0)
# clf.predict("asdfsdfsdgsdgsdg", "KNN", 5,0)
# clf.predict("asdfsdfsdgsdgsdg", "KNN", 7,0)
# clf.predict("asdfsdfsdgsdgsdg", "KNN", 10,0)
# clf.predict("asdfsdfsdgsdgsdg", "KNN", 15,0)
