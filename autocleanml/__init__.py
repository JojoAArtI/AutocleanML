"""
autocleanml
~~~~~~~~~~~
The ML toolkit that teaches you while it works.

Everything is importable from the top-level namespace:
    from autocleanml import guide, run_pipeline, clean, train, ...
"""

from ._guide       import guide
from ._pipeline    import run_pipeline
from ._loader      import load_data, sample_data
from ._cleaner     import clean
from ._splitter    import split
from ._encoder     import encode
from ._scaler      import scale
from ._detector    import detect_task
from ._trainer     import train
from ._comparator  import compare_models
from ._evaluator   import evaluate
from ._predictor   import predict
from ._persistence import save_model, load_model
from ._importance  import get_feature_importance
from ._text        import extract_emails, extract_phones, extract_urls, extract_numbers, extract_dates, extract_all
from ._nlp         import (
    tokenize_text, remove_punctuation, remove_stopwords,
    stem, lemmatize, ngrams, word_freq, clean_text,
    bag_of_words, tfidf, STOPWORDS,
)

__version__ = "1.0.0"
__author__  = "Joel Inian Francis"

__all__ = [
    "guide",
    "run_pipeline",
    "load_data", "sample_data",
    "clean", "split", "encode", "scale",
    "detect_task", "train", "compare_models",
    "evaluate", "predict",
    "save_model", "load_model",
    "get_feature_importance",
    # Text extraction
    "extract_emails", "extract_phones", "extract_urls",
    "extract_numbers", "extract_dates", "extract_all",
    # NLP preprocessing
    "tokenize_text", "remove_punctuation", "remove_stopwords",
    "stem", "lemmatize", "ngrams", "word_freq", "clean_text",
    "bag_of_words", "tfidf", "STOPWORDS",
]
