"""Tests for text extraction and NLP preprocessing modules."""

import pytest
import pandas as pd

from mlguide import (
    extract_emails, extract_phones, extract_urls,
    extract_numbers, extract_dates, extract_all,
    tokenize_text, remove_punctuation, remove_stopwords,
    stem, lemmatize, ngrams, word_freq, clean_text,
    bag_of_words, tfidf, STOPWORDS,
)


# ──────────────────────── Text Extraction ──────────────────────


class TestExtractEmails:
    def test_finds_emails(self):
        text = "Contact hello@example.com and support@site.org for help"
        result = extract_emails(text, verbose=False)
        assert "hello@example.com" in result
        assert "support@site.org" in result

    def test_no_emails(self):
        result = extract_emails("No emails here!", verbose=False)
        assert result == []


class TestExtractPhones:
    def test_finds_phones(self):
        text = "Call me at 555-123-4567 or +1 800 555 1234"
        result = extract_phones(text, verbose=False)
        assert len(result) >= 1


class TestExtractUrls:
    def test_finds_urls(self):
        text = "Visit https://example.com and www.google.com"
        result = extract_urls(text, verbose=False)
        assert "https://example.com" in result


class TestExtractNumbers:
    def test_finds_numbers(self):
        text = "The price is $19.99 and quantity is 42"
        result = extract_numbers(text, verbose=False)
        assert "19.99" in result
        assert "42" in result


class TestExtractDates:
    def test_finds_dates(self):
        text = "Born on 15/03/1999 and graduated 2021-06-15"
        result = extract_dates(text, verbose=False)
        assert len(result) >= 1


class TestExtractAll:
    def test_returns_dict(self):
        text = "Email me at test@mail.com, call 555-1234567, visit https://x.com on 01/01/2025 for $99"
        result = extract_all(text, verbose=False)
        assert isinstance(result, dict)
        assert "emails" in result
        assert "phones" in result
        assert "urls" in result
        assert "numbers" in result
        assert "dates" in result


# ──────────────────────── NLP Preprocessing ────────────────────


class TestTokenize:
    def test_basic_tokenize(self):
        tokens = tokenize_text("Hello World! How are you?")
        assert "hello" in tokens
        assert "world" in tokens

    def test_no_lowercase(self):
        tokens = tokenize_text("Hello World!", lowercase=False)
        assert "Hello" in tokens


class TestRemovePunctuation:
    def test_removes_punctuation(self):
        result = remove_punctuation("Hello, World! How's it?")
        assert "," not in result
        assert "!" not in result


class TestRemoveStopwords:
    def test_removes_common_words(self):
        tokens = ["the", "cat", "is", "on", "the", "mat"]
        result = remove_stopwords(tokens)
        assert "the" not in result
        assert "cat" in result
        assert "mat" in result

    def test_extra_stopwords(self):
        tokens = ["cat", "dog", "fish"]
        result = remove_stopwords(tokens, extra_stopwords=["cat"])
        assert "cat" not in result
        assert "dog" in result


class TestStem:
    def test_basic_stemming(self):
        result = stem(["running", "jumps", "easily"])
        assert len(result) == 3
        # Stems should be shorter than originals
        assert all(len(s) <= len(o) for s, o in zip(result, ["running", "jumps", "easily"]))


class TestLemmatize:
    def test_irregular_verbs(self):
        result = lemmatize(["went", "children", "ran"])
        assert "go" in result
        assert "child" in result
        assert "run" in result


class TestNgrams:
    def test_bigrams(self):
        tokens = ["I", "love", "machine", "learning"]
        result = ngrams(tokens, n=2)
        assert ("I", "love") in result
        assert ("love", "machine") in result
        assert len(result) == 3

    def test_trigrams(self):
        tokens = ["I", "love", "machine", "learning"]
        result = ngrams(tokens, n=3)
        assert len(result) == 2


class TestWordFreq:
    def test_counts(self):
        tokens = ["the", "cat", "the", "dog", "the"]
        result = word_freq(tokens)
        assert isinstance(result, pd.DataFrame)
        assert result.iloc[0]["word"] == "the"
        assert result.iloc[0]["count"] == 3


class TestCleanText:
    def test_single_string(self):
        result = clean_text("The cats were running quickly!", verbose=False)
        assert isinstance(result, str)
        assert "the" not in result.split()  # stopword removed

    def test_list_of_strings(self):
        result = clean_text(["Hello world!", "Goodbye world!"], verbose=False)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_stem_method(self):
        result = clean_text("The runners were running", method="stem", verbose=False)
        assert isinstance(result, str)

    def test_none_method(self):
        result = clean_text("The cat sat", method="none", verbose=False)
        assert isinstance(result, str)


class TestBagOfWords:
    def test_basic_bow(self):
        texts = ["I love ML", "ML is great", "I love great ML"]
        df, vec = bag_of_words(texts, verbose=False)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 3
        assert "ml" in df.columns or "ML" in df.columns.str.upper().tolist()


class TestTfidf:
    def test_basic_tfidf(self):
        texts = ["I love ML", "ML is great", "I love great ML"]
        df, vec = tfidf(texts, verbose=False)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 3
        # TF-IDF values should be non-negative
        assert (df.values >= 0).all()


class TestStopwords:
    def test_stopwords_is_frozenset(self):
        assert isinstance(STOPWORDS, frozenset)
        assert "the" in STOPWORDS
        assert "is" in STOPWORDS
