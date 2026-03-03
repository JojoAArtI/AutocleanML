"""
mlguide._guide
~~~~~~~~~~~~~~
The built-in interactive help system — what makes mlguide different.
"""

__version__ = "1.0.0"
__author__ = "Joel Inian Francis"


# ───────────────────────────────────────────────────────────────
#                       Topic content
# ───────────────────────────────────────────────────────────────

_TOPICS = {}


def _register(key, text):
    _TOPICS[key] = text


# ── Overview (called with no argument) ──
_OVERVIEW = f"""\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  mlguide — The ML toolkit that teaches you.
  v{__version__} | by {__author__}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👋 Welcome! Here's how to use this package.

OPTION 1 — Full autopilot (recommended for beginners):
  from mlguide import run_pipeline
  result = run_pipeline("data.csv", target="price")

OPTION 2 — Use individual steps:
  from mlguide import load_data, clean, split, encode, scale, train, evaluate

  df         = load_data("data.csv")
  df         = clean(df, target="price")
  X_tr, X_te, y_tr, y_te = split(df, target="price")
  X_tr, enc  = encode(X_tr, fit=True)
  X_te, _    = encode(X_te, encoder=enc)
  X_tr, sc   = scale(X_tr, fit=True)
  X_te, _    = scale(X_te, scaler=sc)
  model      = train(X_tr, y_tr)
  metrics    = evaluate(model, X_te, y_te)

OPTION 3 — Compare multiple models:
  from mlguide import compare_models
  leaderboard = compare_models(X_tr, y_tr)

📚 To learn about a specific step:
  guide("clean")      → what data cleaning does and why
  guide("encode")     → what encoding is and why it's needed
  guide("split")      → why we split data and what data leakage is
  guide("train")      → what models are available and when to use each
  guide("evaluate")   → what the metrics mean

🔍 New to ML entirely?
  guide("ml_basics")  → a 2-minute explanation of what ML actually is

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\
"""


# ── ML Basics ──
_register("ml_basics", """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📖 Guide: What is Machine Learning?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Machine Learning (ML) is about teaching a computer to find
  patterns in data so it can make predictions on new, unseen data.

  TWO MAIN TYPES:

  1. REGRESSION  — predicting a number.
     Example: predicting house prices from features like
     square footage, bedrooms, and location.

  2. CLASSIFICATION — predicting a category.
     Example: predicting whether an email is spam or not.

  THE BASIC WORKFLOW:
    ┌────────────┐    ┌────────┐    ┌────────┐
    │  Get data  │ →  │ Clean  │ →  │ Split  │
    └────────────┘    └────────┘    └────────┘
          │
          ▼
    ┌────────────┐    ┌────────┐    ┌──────────┐
    │  Encode &  │ →  │ Train  │ →  │ Evaluate │
    │   Scale    │    │ model  │    │  results │
    └────────────┘    └────────┘    └──────────┘

  mlguide handles every step of this workflow,
  either automatically or one step at a time.

  → guide("workflow") for the full step-by-step diagram
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\
""")


# ── Load ──
_register("load", """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📖 Guide: Loading your data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  load_data() reads a CSV file into a pandas DataFrame,
  or accepts a DataFrame you already have.

  HOW TO USE IT:
    from mlguide import load_data

    df = load_data("data.csv")
    df = load_data("data.csv", target="price")  # validates target exists
    df = load_data(existing_df)                  # works with DataFrames too

  WHAT IT CHECKS:
    • File exists (gives a helpful error if not)
    • Target column exists (suggests close matches if typo)
    • At least 50 rows (ML needs enough data to learn)

  SAMPLE DATASETS:
    Don't have a CSV? Use a bundled sample:
      from mlguide import sample_data
      df = sample_data("regression")       # housing data, target="price"
      df = sample_data("classification")   # labels dataset, target="label"
      df = sample_data("titanic")          # survival, target="Survived"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\
""")


# ── Clean ──
_register("clean", """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📖 Guide: Cleaning your data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Real-world data is messy. clean() handles:

  1. DUPLICATE ROWS — exact copies are removed.
  2. MOSTLY-EMPTY COLUMNS — if >80% of values are missing,
     the column is dropped (threshold is configurable).
  3. HIGH-CARDINALITY COLUMNS — columns with too many unique
     text values (like IDs or names) are dropped because
     they don't help ML models learn general patterns.
  4. MISSING VALUES — numeric columns are filled with the
     median; categorical columns with the most frequent value.

  HOW TO USE IT:
    from mlguide import clean
    df = clean(df, target="price")

  TARGET PROTECTION:
    The target column is never dropped or imputed by accident.

  IMPORTANT: clean() returns a COPY of your data.
    Your original DataFrame is never modified.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\
""")


# ── Split ──
_register("split", """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📖 Guide: Splitting your data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  WHY DO WE SPLIT DATA?
    We split data into a training set and a test set.
    The model learns from the training set.
    The test set is kept completely separate — the model
    never sees it during training.
    We use the test set at the end to check how well the
    model performs on data it has never seen before.

  WHY NOT JUST USE ALL THE DATA?
    If you train on ALL your data and then evaluate on the
    same data, your model will look amazing — but it's just
    memorising, not learning. This is called overfitting.

  ⚠️  IMPORTANT: DATA LEAKAGE
    Data leakage is one of the most common beginner mistakes.
    It happens when information from your test set "leaks"
    into your training process — for example, if you scale
    your features before splitting.
    mlguide prevents this automatically by always
    splitting BEFORE encoding and scaling.

  HOW TO USE IT:
    from mlguide import split
    X_train, X_test, y_train, y_test = split(df, target="price")
    X_train, X_test, y_train, y_test = split(df, target="price", test_size=0.15)

  DEFAULT BEHAVIOUR:
    80% training, 20% test.
    For classification problems, split() automatically uses
    a stratified split to preserve class proportions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\
""")


# ── Encode ──
_register("encode", """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📖 Guide: Encoding your data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  WHAT IS ENCODING?
    Computers only understand numbers. If your data has
    text columns like "red", "blue", "green", we need to
    convert them into numbers before training.

  HOW DOES ONE-HOT ENCODING WORK?
    Each unique value becomes its own column:
    Color          →  Color_blue  Color_green  Color_red
    "red"          →  0           0            1
    "blue"         →  1           0            0
    "green"        →  0           1            0

  WHY FIT ON TRAINING DATA ONLY?
    The encoder learns what categories exist from your
    training data. Test data should be transformed using
    the SAME encoder — to simulate real-world predictions
    where you won't have seen all possible values upfront.

  HOW TO USE IT:
    X_train, enc = encode(X_train, fit=True)
    X_test, _    = encode(X_test, encoder=enc)

  ⚠️  Never re-fit the encoder on test data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\
""")


# ── Scale ──
_register("scale", """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📖 Guide: Scaling your features
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  WHAT IS SCALING?
    Features in your data may have very different ranges.
    For example, "age" might be 18–80 while "income" is
    20,000–200,000. Some models struggle with this.

  METHODS:
    "standard"  → zero mean, unit variance (most common)
    "minmax"    → scales everything to 0–1
    "robust"    → uses median instead of mean (better
                  when your data has outliers)
    "none"      → skip scaling entirely

  WHEN TO SKIP SCALING:
    Tree-based models (RandomForest, GradientBoosting,
    DecisionTree) don't need scaling at all. Linear
    models and KNN benefit from it.

  HOW TO USE IT:
    X_train, sc = scale(X_train, fit=True)
    X_test, _   = scale(X_test, scaler=sc)

  ⚠️  Always fit the scaler on TRAINING data only.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\
""")


# ── Train ──
_register("train", """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📖 Guide: Training a model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  WHAT IS TRAINING?
    Training means showing the model your data and letting
    it learn the patterns. Afterwards, it can make
    predictions on data it has never seen.

  HOW TO CHOOSE A MODEL:
    Not sure? Use model="auto" and let mlguide decide.
    Want to compare all options? Use compare_models().

  REGRESSION (predicting a number):
    "linear"             Simple, fast, interpretable
    "ridge"              Handles correlated features better
    "lasso"              Selects only important features
    "decision_tree"      Easy to visualize and explain
    "random_forest"      Strong default, handles noise
    "gradient_boosting"  Often best accuracy
    "knn"                Simple, good for small datasets

  CLASSIFICATION (predicting a category):
    "logistic_regression" Simple, fast, interpretable
    "decision_tree"       Easy to visualize and explain
    "random_forest"       Strong default
    "gradient_boosting"   Often best accuracy
    "svm"                 Good for smaller datasets
    "knn"                 Simple, intuitive
    "naive_bayes"         Very fast, good for text data

  HOW TO USE IT:
    model = train(X_train, y_train)
    model = train(X_train, y_train, model="random_forest")
    model = train(X_train, y_train, model="gradient_boosting",
                  params={"n_estimators": 200})

  BRING YOUR OWN MODEL:
    from sklearn.neighbors import KNeighborsRegressor
    model = train(X_train, y_train,
                  model=KNeighborsRegressor(n_neighbors=3))

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\
""")


# ── Compare ──
_register("compare", """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📖 Guide: Comparing models
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  WHAT IS CROSS-VALIDATION?
    Instead of using a single train/test split, cross-
    validation splits the training data into K "folds".
    It trains the model K times, each time using a
    different fold as the validation set.

    This gives a more reliable estimate of how well a
    model will perform on unseen data.

  HOW compare_models() WORKS:
    1. Takes your training data (X_train, y_train).
    2. Trains every built-in model using 5-fold CV.
    3. Ranks them by their average CV score.
    4. Returns a leaderboard DataFrame.

  HOW TO USE IT:
    leaderboard = compare_models(X_train, y_train)
    leaderboard = compare_models(X_train, y_train, cv=10)
    leaderboard = compare_models(X_train, y_train,
                    models=["random_forest", "ridge"])

  AFTER COMPARING:
    Pick the best model from the leaderboard and train it:
    model = train(X_train, y_train, model="gradient_boosting")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\
""")


# ── Evaluate ──
_register("evaluate", """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📖 Guide: Understanding evaluation metrics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  REGRESSION METRICS:

    RMSE (Root Mean Squared Error)
      Average prediction error, in the same units as your
      target. Lower is better. Penalises big mistakes.

    MAE (Mean Absolute Error)
      Average absolute error. Easier to interpret than
      RMSE. Lower is better.

    R² (R-squared)
      How much of the variation in the target your model
      explains. 1.0 = perfect. 0.0 = no better than
      predicting the mean. Can be negative = very bad.

  CLASSIFICATION METRICS:

    Accuracy
      Fraction of correct predictions. Can be misleading
      if classes are imbalanced (e.g. 99% class A).

    Precision
      Of all items predicted as positive, how many actually
      were positive?

    Recall
      Of all actual positives, how many were found?

    F1 Score
      Harmonic mean of precision and recall. Balances both.

  HOW TO USE IT:
    metrics = evaluate(model, X_test, y_test)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\
""")


# ── Predict ──
_register("predict", """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📖 Guide: Making predictions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  USING A LIVE MODEL:
    from mlguide import predict
    preds = predict(model, X_test)

  USING A SAVED MODEL:
    preds = predict("my_model.pkl", "new_data.csv")
    preds = predict("my_model.pkl", new_dataframe)

    When you use a saved .pkl file, predict() automatically
    loads the encoder and scaler and applies the same
    preprocessing to your new data.

  CLASSIFICATION PROBABILITIES:
    probs = predict(model, X_test, return_proba=True)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\
""")


# ── Save ──
_register("save", """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📖 Guide: Saving and loading models
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  SAVING:
    from mlguide import save_model
    save_model(model, "my_model.pkl",
               encoder=enc, scaler=sc,
               metadata={"target": "price"})

    This saves the model, encoder, and scaler together.
    This means you can use the model later without
    needing to remember how you preprocessed the data.

  LOADING:
    from mlguide import load_model
    bundle = load_model("my_model.pkl")
    bundle["model"]     # the fitted model
    bundle["encoder"]   # the fitted encoder
    bundle["scaler"]    # the fitted scaler
    bundle["metadata"]  # any extra info you stored

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\
""")


# ── Workflow ──
_register("workflow", """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📖 Guide: The recommended workflow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Step 1: LOAD your data
    df = load_data("data.csv")

  Step 2: CLEAN it
    df = clean(df, target="price")

  Step 3: SPLIT into train / test
    X_tr, X_te, y_tr, y_te = split(df, target="price")

  Step 4: ENCODE categorical columns
    X_tr, enc = encode(X_tr, fit=True)
    X_te, _   = encode(X_te, encoder=enc)

  Step 5: SCALE numeric features
    X_tr, sc = scale(X_tr, fit=True)
    X_te, _  = scale(X_te, scaler=sc)

  Step 6: TRAIN a model
    model = train(X_tr, y_tr)

  Step 7: EVALUATE on test data
    metrics = evaluate(model, X_te, y_te)

  Step 8 (optional): SAVE for later use
    save_model(model, "model.pkl", encoder=enc, scaler=sc)

  ⚠️  CRITICAL ORDER:
    Split BEFORE encode and scale.
    Fit encoders/scalers on TRAINING data only.
    This prevents data leakage.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\
""")


# ── Leakage ──
_register("leakage", """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📖 Guide: Data Leakage — the invisible mistake
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  WHAT IS DATA LEAKAGE?
    Data leakage happens when information from your test
    set accidentally influences your training process.
    Your model appears to work brilliantly, but in the
    real world it fails.

  THE MOST COMMON CAUSE:
    Fitting a scaler or encoder on the ENTIRE dataset
    before splitting.

  EXAMPLE OF WHAT GOES WRONG:

    ❌ WRONG (leakage):
      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(X)     # fit on ALL data
      X_train, X_test = train_test_split(X_scaled)

    ✅ CORRECT (no leakage):
      X_train, X_test = train_test_split(X)
      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_train)  # fit on TRAIN only
      X_test  = scaler.transform(X_test)        # transform TEST

  WHY DOES THIS MATTER?
    When you fit on all data, the scaler learns the mean
    and standard deviation of the test set too. The test
    set is supposed to be completely unknown to the model.

  HOW mlguide PROTECTS YOU:
    • split() is called BEFORE encode() and scale().
    • encode() and scale() have a fit/reuse pattern that
      makes the correct order natural.
    • If you encode a suspiciously large dataset, a
      PreprocessingOrderWarning is emitted.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\
""")


# ── Text Extraction ──
_register("text", """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📖 Guide: Text Extraction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Tired of writing regex? mlguide has you covered.
  Extract emails, phones, URLs, numbers, and dates from
  any text — no regex knowledge needed.

  EXTRACT EMAILS:
    from mlguide import extract_emails
    extract_emails("Contact hello@example.com")
    # ['hello@example.com']

  EXTRACT PHONE NUMBERS:
    from mlguide import extract_phones
    extract_phones("Call +1 555-123-4567")
    # ['+1 555-123-4567']

  EXTRACT URLS:
    from mlguide import extract_urls
    extract_urls("Visit https://example.com")
    # ['https://example.com']

  EXTRACT NUMBERS:
    from mlguide import extract_numbers
    extract_numbers("Price is $19.99, qty 42")
    # ['19.99', '42']

  EXTRACT DATES:
    from mlguide import extract_dates
    extract_dates("Born 15/03/1999")
    # ['15/03/1999']

  EXTRACT EVERYTHING AT ONCE:
    from mlguide import extract_all
    result = extract_all(paragraph)
    result["emails"]   # all emails
    result["phones"]   # all phone numbers
    result["urls"]     # all URLs
    result["numbers"]  # all numbers
    result["dates"]    # all dates

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\
""")


# ── NLP Preprocessing ──
_register("nlp", """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📖 Guide: NLP Preprocessing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  NLP = Natural Language Processing — teaching computers
  to understand human text.

  WHAT IS PREPROCESSING?
    Raw text is messy. Before ML can use it, we need to:
    1. Lowercase everything
    2. Remove punctuation
    3. Split into words (tokenise)
    4. Remove unimportant words (stopwords)
    5. Reduce words to their root (stem or lemmatise)
    6. Convert to numbers (vectorise)

  TOKENISE:
    from mlguide import tokenize_text
    tokenize_text("Hello World!")
    # ['hello', 'world']

  REMOVE STOPWORDS:
    from mlguide import remove_stopwords
    remove_stopwords(["the", "cat", "is", "on", "the", "mat"])
    # ['cat', 'mat']

  STEM (reduce to root form):
    from mlguide import stem
    stem(["running", "jumps", "easily"])
    # ['run', 'jump', 'easili']

  LEMMATISE (smarter root form):
    from mlguide import lemmatize
    lemmatize(["running", "children", "went"])
    # ['run', 'child', 'go']

  FULL PIPELINE IN ONE CALL:
    from mlguide import clean_text
    clean_text("The cats were running quickly!")
    # 'cat run quick'

  VECTORISE (convert text to numbers for ML):
    from mlguide import bag_of_words, tfidf
    bow_df, vec = bag_of_words(["I love ML", "ML is great"])
    tfidf_df, vec = tfidf(["I love ML", "ML is great"])

  N-GRAMS:
    from mlguide import ngrams
    ngrams(["I", "love", "machine", "learning"], n=2)
    # [('I', 'love'), ('love', 'machine'), ...]

  WORD FREQUENCIES:
    from mlguide import word_freq
    word_freq(["the", "cat", "the", "dog"])

  ⚠️  No NLTK or spaCy required — everything is built-in!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\
""")


# ── Cheatsheet ──
_register("cheatsheet", """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📖 mlguide Cheatsheet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  LOADING:
    df = load_data("data.csv", target="price")
    df = sample_data("regression")

  CLEANING:
    df = clean(df, target="price")

  SPLITTING:
    X_tr, X_te, y_tr, y_te = split(df, target="price")

  ENCODING:
    X_tr, enc = encode(X_tr, fit=True)
    X_te, _   = encode(X_te, encoder=enc)

  SCALING:
    X_tr, sc  = scale(X_tr, fit=True)
    X_te, _   = scale(X_te, scaler=sc)

  TRAINING:
    model = train(X_tr, y_tr, model="gradient_boosting")
    leaderboard = compare_models(X_tr, y_tr)

  EVALUATING:
    metrics = evaluate(model, X_te, y_te)

  SAVING / LOADING:
    save_model(model, "model.pkl", encoder=enc, scaler=sc)
    bundle = load_model("model.pkl")

  TEXT EXTRACTION:
    extract_emails(text)    extract_phones(text)
    extract_urls(text)      extract_numbers(text)
    extract_dates(text)     extract_all(text)

  NLP PREPROCESSING:
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)  OR  lemmatize(tokens)
    cleaned = clean_text(text)
    bow_df, vec = bag_of_words(texts)
    tfidf_df, vec = tfidf(texts)
    grams = ngrams(tokens, n=2)
    freq = word_freq(tokens)

  FULL AUTOPILOT:
    result = run_pipeline("data.csv", target="price")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\
""")


# ───────────────────────────────────────────────────────────────
#                       Public API
# ───────────────────────────────────────────────────────────────

def guide(topic=None):
    """Interactive help system for mlguide.

    Parameters
    ----------
    topic : str, optional
        Topic to explain.  ``None`` prints the overview.
        Available: ``"ml_basics"``, ``"load"``, ``"clean"``,
        ``"split"``, ``"encode"``, ``"scale"``, ``"train"``,
        ``"compare"``, ``"evaluate"``, ``"predict"``, ``"save"``,
        ``"workflow"``, ``"leakage"``, ``"text"``, ``"nlp"``,
        ``"cheatsheet"``.
    """
    if topic is None:
        print(_OVERVIEW)
        return

    key = topic.lower().strip()
    if key in _TOPICS:
        print(_TOPICS[key])
    else:
        available = ", ".join(sorted(_TOPICS.keys()))
        print(f'❓ Unknown guide topic: "{topic}"')
        print(f"   Available topics: {available}")
        print(f'   Or call guide() with no arguments for the overview.')
