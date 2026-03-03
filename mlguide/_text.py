"""
autocleanml._text
~~~~~~~~~~~~~~~~~
Regex-based text extraction utilities.

Extract emails, phone numbers, URLs, dates, and numbers from any string
so students never have to memorise regex patterns again.
"""

import re


# ───────────────────────── Compiled patterns ──────────────────────────

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

_PHONE_RE = re.compile(
    r"(?:"
    r"\+?\d{1,3}[\s\-.]?)?"            # optional country code
    r"(?:\(?\d{2,4}\)?[\s\-.]?)?"      # optional area code
    r"\d{3,5}[\s\-.]?\d{3,5}"          # main number
)

_URL_RE = re.compile(
    r"https?://[^\s<>\"']+|"
    r"www\.[^\s<>\"']+"
)

_NUMBER_RE = re.compile(
    r"-?\d+(?:,\d{3})*(?:\.\d+)?"
)

_DATE_RE = re.compile(
    r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b|"             # DD/MM/YYYY etc.
    r"\b\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\b|"                # YYYY-MM-DD
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"[\s,]+\d{1,2}(?:[\s,]+\d{2,4})?\b",                    # Month DD, YYYY
    re.IGNORECASE,
)


# ───────────────────────── Public API ─────────────────────────────────


def extract_emails(text, verbose=True):
    """Find all email addresses in a string.

    Parameters
    ----------
    text : str
        Input text.
    verbose : bool
        Print results summary.

    Returns
    -------
    list[str]

    Example
    -------
    >>> extract_emails("Contact us at hello@example.com or support@site.org")
    ['hello@example.com', 'support@site.org']
    """
    results = _EMAIL_RE.findall(text)
    if verbose:
        print(f"\n📧 Found {len(results)} email(s): {results}")
    return results


def extract_phones(text, verbose=True):
    """Find all phone numbers in a string.

    Parameters
    ----------
    text : str
        Input text.
    verbose : bool
        Print results summary.

    Returns
    -------
    list[str]

    Example
    -------
    >>> extract_phones("Call me at +1 555-123-4567 or (044) 98765 43210")
    ['+1 555-123-4567', '(044) 98765 43210']
    """
    results = [m.strip() for m in _PHONE_RE.findall(text) if len(m.strip()) >= 7]
    if verbose:
        print(f"\n📞 Found {len(results)} phone number(s): {results}")
    return results


def extract_urls(text, verbose=True):
    """Find all URLs in a string.

    Parameters
    ----------
    text : str
        Input text.
    verbose : bool
        Print results summary.

    Returns
    -------
    list[str]

    Example
    -------
    >>> extract_urls("Visit https://example.com or www.google.com")
    ['https://example.com', 'www.google.com']
    """
    results = _URL_RE.findall(text)
    if verbose:
        print(f"\n🔗 Found {len(results)} URL(s): {results}")
    return results


def extract_numbers(text, verbose=True):
    """Find all numbers (integers and decimals) in a string.

    Parameters
    ----------
    text : str
        Input text.
    verbose : bool
        Print results summary.

    Returns
    -------
    list[str]

    Example
    -------
    >>> extract_numbers("The price is $19.99 and quantity is 42")
    ['19.99', '42']
    """
    results = _NUMBER_RE.findall(text)
    if verbose:
        print(f"\n🔢 Found {len(results)} number(s): {results}")
    return results


def extract_dates(text, verbose=True):
    """Find all date-like patterns in a string.

    Parameters
    ----------
    text : str
        Input text.
    verbose : bool
        Print results summary.

    Returns
    -------
    list[str]

    Example
    -------
    >>> extract_dates("Born on 15/03/1999 and graduated March 20, 2021")
    ['15/03/1999', 'March 20, 2021']
    """
    results = [m.strip() for m in _DATE_RE.findall(text)]
    if verbose:
        print(f"\n📅 Found {len(results)} date(s): {results}")
    return results


def extract_all(text, verbose=True):
    """Extract everything at once — emails, phones, URLs, numbers, dates.

    Parameters
    ----------
    text : str
        Input text.
    verbose : bool
        Print results summary.

    Returns
    -------
    dict
        Keys: ``emails``, ``phones``, ``urls``, ``numbers``, ``dates``.

    Example
    -------
    >>> result = extract_all("Email me at test@mail.com, call +1-555-1234, visit https://x.com on 01/01/2025")
    >>> result["emails"]
    ['test@mail.com']
    """
    result = {
        "emails": extract_emails(text, verbose=False),
        "phones": extract_phones(text, verbose=False),
        "urls": extract_urls(text, verbose=False),
        "numbers": extract_numbers(text, verbose=False),
        "dates": extract_dates(text, verbose=False),
    }
    if verbose:
        total = sum(len(v) for v in result.values())
        print(f"\n🔍 Extracted {total} items from text:")
        for key, vals in result.items():
            if vals:
                print(f"   {key:>8}: {vals}")
    return result
