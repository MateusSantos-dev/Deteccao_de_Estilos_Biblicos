from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


def create_tfidf_vect(texts: list[str],
                      ngrams: tuple[int, int] = (1, 2),
                      max_features: int = 10000,
                      min_df: int = 2,
                      max_df: float = 0.8,
                      use_idf: bool = True,
                      sublinear_tf: bool = True
                      ) -> tuple[csr_matrix, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        ngram_range=ngrams,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        use_idf=use_idf,
        sublinear_tf=sublinear_tf,
        lowercase=True,
        strip_accents='unicode'
    )

    try:
        matrix = vectorizer.fit_transform(texts)
        return matrix, vectorizer
    except Exception as e:
        raise ValueError(f"Erro na vetorização tfidf: {e}")
