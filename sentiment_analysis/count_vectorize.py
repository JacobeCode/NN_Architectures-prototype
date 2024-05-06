from sklearn.feature_extraction.text import CountVectorizer

def count_vectorize(sample, mode):
    count_vector = CountVectorizer()
    if mode == "train":
        counts = count_vector.fit_transform(sample)
    elif mode == "test":
        counts = count_vector.transform(sample)
    else:
        raise NameError
    return counts