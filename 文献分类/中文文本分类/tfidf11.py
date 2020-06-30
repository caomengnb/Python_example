from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from Tools import readfile, readbunchobj, writebunchobj

def vector_space(stopword_path, bunch_path, space_path, train_tfidf_path=None):
    stpwrdlst = readfile(stopword_path).splitlines()
    bunch = readbunchobj(bunch_path)
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})

    if train_tfidf_path is not None:
        trainbunch = readbunchobj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,
                                     vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)

    else:
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_

    writebunchobj(space_path, tfidfspace)
    print("if-idf词向量空间创建完成！")


def tezhen():
    stopword_path = "trainxl/hlt_stop_words.txt"
    bunch_path = "trainxl/trainxl.dat"
    space_path = "trainxl/traintfdif.dat"
    vector_space(stopword_path, bunch_path, space_path)

    bunch_path = "testxl/testxl.dat"
    space_path = "testxl/testtfidf.dat"
    train_tfidf_path = "trainxl/traintfdif.dat"
    vector_space(stopword_path, bunch_path, space_path, train_tfidf_path)