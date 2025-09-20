from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix


def create_bag_of_words_vect(texts: list[str], ngram: tuple[int, int] = (1, 1)) -> tuple[csr_matrix, CountVectorizer]:
    vectorizer = CountVectorizer(ngram_range=ngram)
    word_dict = vectorizer.fit_transform(texts)
    return word_dict, vectorizer


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
