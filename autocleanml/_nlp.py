"""
autocleanml._nlp
~~~~~~~~~~~~~~~~
NLP preprocessing — tokenisation, stopword removal, stemming,
lemmatisation, and vectorisation.

Zero external NLP dependencies — uses only stdlib + scikit-learn.
"""

import re
import string
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ───────────────────────── Built-in stopwords ─────────────────────────
# Comprehensive English stopword list — no NLTK required.

STOPWORDS = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o",
    "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn",
    "also", "could", "would", "shall", "may", "might", "must", "need",
    "ever", "every", "still", "already", "even", "much", "many", "well",
    "back", "like", "since", "get", "got", "go", "went", "gone",
    "come", "came", "make", "made", "take", "took", "give", "gave",
    "say", "said", "tell", "told", "know", "knew", "think", "thought",
    "see", "saw", "want", "let", "keep", "kept", "put",
})


# ───────────────────── Porter Stemmer (built-in) ─────────────────────
# Simplified Porter stemmer — no NLTK needed.

def _porter_step1(word):
    """Step 1: plurals & past participles."""
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    if word.endswith("eed"):
        return word[:-1] if len(word) > 4 else word
    for suffix in ("ed", "ing"):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if any(c in "aeiou" for c in stem) and len(stem) >= 2:
                if stem.endswith("at") or stem.endswith("bl") or stem.endswith("iz"):
                    return stem + "e"
                if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
                    return stem[:-1]
                return stem
            return word
    return word


def _porter_step2(word):
    """Step 2: common suffixes."""
    replacements = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
        ("alli", "al"), ("entli", "ent"), ("eli", "e"),
        ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
        ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
        ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
        ("iviti", "ive"), ("biliti", "ble"),
    ]
    for suffix, replacement in replacements:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            return word[: -len(suffix)] + replacement
    return word


def _porter_step3(word):
    """Step 3: more suffixes."""
    replacements = [
        ("icate", "ic"), ("ative", ""), ("alize", "al"),
        ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""),
    ]
    for suffix, replacement in replacements:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            return word[: -len(suffix)] + replacement
    return word


def _porter_stem_word(word):
    """Apply simplified Porter stemming to a single word."""
    if len(word) <= 2:
        return word
    word = _porter_step1(word)
    word = _porter_step2(word)
    word = _porter_step3(word)
    # Final cleanup
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


# ─────────────────────── Simple Lemmatizer ───────────────────────────
# Rule-based lemmatizer — handles common English patterns without NLTK.

_IRREGULAR_VERBS = {
    "ran": "run", "was": "be", "were": "be", "been": "be",
    "had": "have", "has": "have", "did": "do", "does": "do",
    "went": "go", "gone": "go", "saw": "see", "seen": "see",
    "took": "take", "taken": "take", "gave": "give", "given": "give",
    "made": "make", "came": "come", "said": "say", "told": "tell",
    "knew": "know", "known": "know", "thought": "think",
    "found": "find", "got": "get", "gotten": "get",
    "left": "leave", "felt": "feel", "kept": "keep",
    "brought": "bring", "began": "begin", "begun": "begin",
    "wrote": "write", "written": "write", "ate": "eat", "eaten": "eat",
    "spoke": "speak", "spoken": "speak", "broke": "break", "broken": "break",
    "chose": "choose", "chosen": "choose", "drove": "drive", "driven": "drive",
    "fell": "fall", "fallen": "fall", "flew": "fly", "flown": "fly",
    "grew": "grow", "grown": "grow", "hid": "hide", "hidden": "hide",
    "held": "hold", "lay": "lie", "led": "lead", "lost": "lose",
    "met": "meet", "paid": "pay", "read": "read", "rode": "ride",
    "ridden": "ride", "rose": "rise", "risen": "rise", "sat": "sit",
    "sold": "sell", "sent": "send", "shook": "shake", "shaken": "shake",
    "sang": "sing", "sung": "sing", "sank": "sink", "sunk": "sink",
    "slept": "sleep", "stood": "stand", "stole": "steal", "stolen": "steal",
    "swam": "swim", "swum": "swim", "taught": "teach", "threw": "throw",
    "thrown": "throw", "understood": "understand", "woke": "wake",
    "woken": "wake", "wore": "wear", "worn": "wear", "won": "win",
    "children": "child", "men": "man", "women": "woman",
    "teeth": "tooth", "feet": "foot", "geese": "goose",
    "mice": "mouse", "people": "person", "oxen": "ox",
}

_IRREGULAR_PLURALS = {
    "analyses": "analysis", "theses": "thesis", "crises": "crisis",
    "phenomena": "phenomenon", "criteria": "criterion", "data": "datum",
    "alumni": "alumnus", "fungi": "fungus", "cacti": "cactus",
    "indices": "index", "matrices": "matrix", "vertices": "vertex",
}


def _lemmatize_word(word):
    """Rule-based lemmatisation for a single word."""
    if len(word) <= 2:
        return word

    lower = word.lower()

    # Check irregular forms
    if lower in _IRREGULAR_VERBS:
        return _IRREGULAR_VERBS[lower]
    if lower in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[lower]

    # Verb forms
    if lower.endswith("ies") and len(lower) > 4:
        return lower[:-3] + "y"
    if lower.endswith("ves") and len(lower) > 4:
        return lower[:-3] + "f"
    if lower.endswith("ing") and len(lower) > 5:
        stem = lower[:-3]
        if stem.endswith(stem[-1]) and stem[-1] not in "aeiou" and len(stem) > 2:
            return stem[:-1]
        return stem + "e" if not stem.endswith("e") else stem
    if lower.endswith("ed") and len(lower) > 4:
        stem = lower[:-2]
        if stem.endswith(stem[-1]) and stem[-1] not in "aeiou" and len(stem) > 2:
            return stem[:-1]
        if not any(c in "aeiou" for c in stem):
            return lower  # probably not a verb
        return stem + "e" if not stem.endswith("e") else stem
    if lower.endswith("ly") and len(lower) > 4:
        return lower[:-2]
    if lower.endswith("ment") and len(lower) > 6:
        return lower[:-4]
    if lower.endswith("ness") and len(lower) > 5:
        return lower[:-4]

    # Noun plurals
    if lower.endswith("sses"):
        return lower[:-2]
    if lower.endswith("ies") and len(lower) > 4:
        return lower[:-3] + "y"
    if lower.endswith("es") and len(lower) > 3:
        if lower.endswith("shes") or lower.endswith("ches") or lower.endswith("xes") or lower.endswith("zes"):
            return lower[:-2]
        return lower[:-1]
    if lower.endswith("s") and not lower.endswith("ss") and len(lower) > 3:
        return lower[:-1]

    return lower


# ───────────────────────── Public API ─────────────────────────────────


def tokenize_text(text, lowercase=True):
    """Split text into word tokens.

    Parameters
    ----------
    text : str
        Input text.
    lowercase : bool
        Convert to lowercase.

    Returns
    -------
    list[str]

    Example
    -------
    >>> tokenize_text("Hello World! How's it going?")
    ['hello', 'world', "how's", 'it', 'going']
    """
    if lowercase:
        text = text.lower()
    # Keep apostrophes within words, split on everything else
    tokens = re.findall(r"\b[a-zA-Z']+\b", text)
    # Remove standalone apostrophes
    tokens = [t.strip("'") for t in tokens if t.strip("'")]
    return tokens


def remove_punctuation(text):
    """Remove all punctuation from text.

    Parameters
    ----------
    text : str

    Returns
    -------
    str

    Example
    -------
    >>> remove_punctuation("Hello, World! How are you?")
    'Hello World How are you'
    """
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_stopwords(tokens, extra_stopwords=None):
    """Remove common English stopwords from a token list.

    Parameters
    ----------
    tokens : list[str]
        Word tokens (lowercase recommended).
    extra_stopwords : list[str], optional
        Additional words to remove.

    Returns
    -------
    list[str]

    Example
    -------
    >>> remove_stopwords(["the", "cat", "is", "on", "the", "mat"])
    ['cat', 'mat']
    """
    stop = STOPWORDS
    if extra_stopwords:
        stop = stop | set(w.lower() for w in extra_stopwords)
    return [t for t in tokens if t.lower() not in stop]


def stem(tokens):
    """Apply Porter stemming to a list of tokens.

    Parameters
    ----------
    tokens : list[str]

    Returns
    -------
    list[str]

    Example
    -------
    >>> stem(["running", "jumps", "easily", "connected"])
    ['run', 'jump', 'easili', 'connect']
    """
    return [_porter_stem_word(t) for t in tokens]


def lemmatize(tokens):
    """Rule-based lemmatisation — no NLTK required.

    Handles common English verb forms, plurals, and irregular words.

    Parameters
    ----------
    tokens : list[str]

    Returns
    -------
    list[str]

    Example
    -------
    >>> lemmatize(["running", "children", "went", "better"])
    ['run', 'child', 'go', 'better']
    """
    return [_lemmatize_word(t) for t in tokens]


def ngrams(tokens, n=2):
    """Generate n-grams from a list of tokens.

    Parameters
    ----------
    tokens : list[str]
    n : int
        n-gram size (2 = bigrams, 3 = trigrams, etc.)

    Returns
    -------
    list[tuple[str]]

    Example
    -------
    >>> ngrams(["I", "love", "machine", "learning"], n=2)
    [('I', 'love'), ('love', 'machine'), ('machine', 'learning')]
    """
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def word_freq(tokens, top_n=None):
    """Count word frequencies.

    Parameters
    ----------
    tokens : list[str]
    top_n : int, optional
        Return only the top N most common words.

    Returns
    -------
    pd.DataFrame
        Columns: ``word``, ``count``.
    """
    counts = Counter(tokens)
    items = counts.most_common(top_n)
    return pd.DataFrame(items, columns=["word", "count"])


def clean_text(text, lowercase=True, remove_stops=True, method="lemmatize",
               extra_stopwords=None, verbose=True):
    """Full NLP preprocessing pipeline in one call.

    Steps: lowercase → remove punctuation → tokenise → remove stopwords →
    stem or lemmatise → rejoin.

    Parameters
    ----------
    text : str or list[str]
        A single string or a list of strings.
    lowercase : bool
        Convert to lowercase.
    remove_stops : bool
        Remove stopwords.
    method : str
        ``"lemmatize"``, ``"stem"``, or ``"none"``.
    extra_stopwords : list[str], optional
        Additional stopwords.
    verbose : bool
        Print summary.

    Returns
    -------
    str or list[str]

    Example
    -------
    >>> clean_text("The cats were running quickly towards the doors!")
    'cat run quick toward door'
    """
    single = isinstance(text, str)
    texts = [text] if single else list(text)

    results = []
    for t in texts:
        # 1. Remove punctuation
        t = remove_punctuation(t)
        # 2. Tokenise
        tokens = tokenize_text(t, lowercase=lowercase)
        # 3. Remove stopwords
        if remove_stops:
            tokens = remove_stopwords(tokens, extra_stopwords)
        # 4. Stem or Lemmatise
        if method == "stem":
            tokens = stem(tokens)
        elif method == "lemmatize":
            tokens = lemmatize(tokens)
        results.append(" ".join(tokens))

    if verbose:
        n = len(results)
        avg_len = sum(len(r.split()) for r in results) / max(n, 1)
        print(f"\n📝 Cleaned {n} text(s) — avg {avg_len:.0f} words after preprocessing")
        if n == 1:
            preview = results[0][:100]
            print(f'   Preview: "{preview}{"..." if len(results[0]) > 100 else ""}"')
        print(f'   → guide("nlp") to learn about NLP preprocessing')

    return results[0] if single else results


def bag_of_words(texts, max_features=None, verbose=True):
    """Convert texts to a Bag-of-Words matrix (CountVectorizer).

    Parameters
    ----------
    texts : list[str]
        List of documents.
    max_features : int, optional
        Max vocabulary size.
    verbose : bool
        Print summary.

    Returns
    -------
    tuple
        ``(feature_matrix_DataFrame, fitted_vectorizer)``

    Example
    -------
    >>> bow_df, vec = bag_of_words(["I love ML", "ML is great"])
    """
    vec = CountVectorizer(max_features=max_features)
    matrix = vec.fit_transform(texts)
    df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names_out())

    if verbose:
        print(f"\n📊 Bag of Words: {df.shape[0]} documents × {df.shape[1]} features")
        print(f'   → Use tfidf() for weighted version')

    return df, vec


def tfidf(texts, max_features=None, verbose=True):
    """Convert texts to a TF-IDF matrix.

    Parameters
    ----------
    texts : list[str]
        List of documents.
    max_features : int, optional
        Max vocabulary size.
    verbose : bool
        Print summary.

    Returns
    -------
    tuple
        ``(tfidf_matrix_DataFrame, fitted_vectorizer)``

    Example
    -------
    >>> tfidf_df, vec = tfidf(["I love ML", "ML is great"])
    """
    vec = TfidfVectorizer(max_features=max_features)
    matrix = vec.fit_transform(texts)
    df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names_out())

    if verbose:
        print(f"\n📊 TF-IDF: {df.shape[0]} documents × {df.shape[1]} features")

    return df, vec
