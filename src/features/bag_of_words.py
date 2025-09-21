from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from typing import Union


def create_bag_of_words_vect(
        texts: list[str],
        ngrams: tuple[int, int] = (1, 1),
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        max_features: int = None
) -> tuple[csr_matrix, CountVectorizer]:
    vectorizer = CountVectorizer(ngram_range=ngrams, min_df=min_df, max_df=max_df, max_features=max_features)
    try:
        word_dict = vectorizer.fit_transform(texts)
        return word_dict, vectorizer
    except Exception as e:
        raise ValueError(f"Erro na vetorização Bag of Words: {e}")


def transform_bag_of_words_vect(texts: list[str], vectorizer: CountVectorizer) -> csr_matrix:
    return vectorizer.transform(texts)


def teste():
    from src.data.load import load_data

    df = load_data("train_arcaico_moderno.csv")
    print(df.columns)
    word_dict, vec = create_bag_of_words_vect(df.text)

    print(word_dict)


if __name__ == "__main__":
    teste()
