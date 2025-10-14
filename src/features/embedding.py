import numpy as np
from gensim.models import Word2Vec


def create_word2vec_features(texts: list[str],
                             vector_size: int = 100,
                             window: int = 5,
                             min_count: int = 2,
                             workers: int = 4,
                             sg: int = 0
                             ) -> np.ndarray:

    tokenized_texts = [text.lower().split() for text in texts]

    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg
    )

    features = []
    for tokens in tokenized_texts:
        word_vectors = []
        for word in tokens:
            if word in model.wv:
                word_vectors.append(model.wv[word])

        if word_vectors:
            doc_vector = np.mean(word_vectors, axis=0)
        else:
            doc_vector = np.zeros(vector_size)

        features.append(doc_vector)

    return np.array(features)
