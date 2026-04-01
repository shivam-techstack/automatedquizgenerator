"""
text_processing.py
------------------
Core NLP pipeline module for the Automated Quiz Generator.

This module implements the fundamental NLP steps:
  1. Text Cleaning
  2. Sentence Tokenization
  3. Word Tokenization
  4. Stopword Removal
  5. Keyword Extraction using Word Frequency (TF-based)

All methods use rule-based logic — no pretrained models or APIs.
"""

import re
import os
import string
from collections import Counter


# ─────────────────────────────────────────────
#  LOAD STOPWORDS FROM FILE
# ─────────────────────────────────────────────

def load_stopwords():
    """
    Load stopwords from the dataset/stopwords.txt file.
    Returns a set of lowercase stopword strings.
    """
    stopwords_path = os.path.join(
        os.path.dirname(__file__), '..', 'dataset', 'stopwords.txt'
    )
    stopwords = set()
    if os.path.exists(stopwords_path):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    stopwords.add(word)
    return stopwords


STOPWORDS = load_stopwords()


# ─────────────────────────────────────────────
#  1. CLEAN TEXT
# ─────────────────────────────────────────────

def clean_text(text):
    """
    Clean the input text by:
    - Removing extra whitespace and newlines
    - Removing special characters except basic punctuation
    - Normalizing multiple spaces to a single space

    Algorithm:
      Input text → strip extra spaces → normalize newlines → remove odd chars
    """
    # Normalize line breaks to spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Remove characters that are not alphanumeric, space, or sentence punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:\'\"\-]', ' ', text)
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─────────────────────────────────────────────
#  2. SENTENCE TOKENIZATION
# ─────────────────────────────────────────────

def tokenize_sentences(text):
    """
    Split text into individual sentences using rule-based boundary detection.

    Algorithm:
      - Split on '.', '!', '?' followed by a space and uppercase letter
        (classic sentence-ending punctuation pattern)
      - Use a regex lookbehind so the punctuation stays with the sentence
      - Filter out empty or very short results

    This avoids false splits on abbreviations like "Dr." or "U.S." to some extent
    by requiring the next char to be uppercase.
    """
    # Split on sentence-ending punctuation followed by space + capital or end of string
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(text)

    # Clean and filter
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences


# ─────────────────────────────────────────────
#  3. WORD TOKENIZATION
# ─────────────────────────────────────────────

def tokenize_words(text):
    """
    Tokenize a string into individual words.

    Algorithm:
      - Lowercase the text
      - Use regex to extract only alphabetic tokens (skip numbers, punctuation)
      - Returns a flat list of word strings
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words


# ─────────────────────────────────────────────
#  4. STOPWORD REMOVAL
# ─────────────────────────────────────────────

def remove_stopwords(words):
    """
    Filter out common stopwords from a list of words.

    Algorithm:
      - Compare each word (lowercased) against the STOPWORDS set
      - Return only words NOT in the stopword set
      - Also remove single-character words

    Stopwords are loaded from dataset/stopwords.txt
    """
    return [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]


# ─────────────────────────────────────────────
#  5. KEYWORD EXTRACTION (Word Frequency / TF)
# ─────────────────────────────────────────────

def extract_keywords_using_frequency(text, top_n=15):
    """
    Extract the most important keywords from text using word frequency.

    Algorithm (Term Frequency based):
      1. Tokenize text into words
      2. Remove stopwords
      3. Count frequency of each remaining word using Counter
      4. Sort by frequency (descending)
      5. Return top_n most frequent words as keywords

    This is a simplified TF (Term Frequency) approach:
      TF(word) = count(word) / total_words

    Since we're ranking relative to each other, raw counts suffice.
    """
    words = tokenize_words(text)
    filtered = remove_stopwords(words)

    if not filtered:
        return []

    freq = Counter(filtered)
    # Return only the word strings (not counts), top_n results
    keywords = [word for word, count in freq.most_common(top_n)]
    return keywords


# ─────────────────────────────────────────────
#  6. RULE-BASED POS TAGGING (Simple)
# ─────────────────────────────────────────────

def simple_pos_tag(word):
    """
    Assign a simple Part-of-Speech tag using rule-based heuristics.

    Rules:
      - Ends in 'ing'  → VERB (present participle)
      - Ends in 'ed'   → VERB (past tense)
      - Ends in 'ly'   → ADV  (adverb)
      - Ends in 'tion' or 'ness' or 'ment' or 'ity' → NOUN
      - Ends in 'al' or 'ous' or 'ive' or 'ic'      → ADJ
      - Otherwise      → NOUN (default assumption for content words)

    This is intentionally simplified for educational purposes.
    """
    w = word.lower()
    if w.endswith('ing'):
        return 'VERB'
    elif w.endswith('ed'):
        return 'VERB'
    elif w.endswith('ly'):
        return 'ADV'
    elif w.endswith(('tion', 'ness', 'ment', 'ity', 'ism', 'ist')):
        return 'NOUN'
    elif w.endswith(('al', 'ous', 'ive', 'ic', 'ful', 'less')):
        return 'ADJ'
    else:
        return 'NOUN'  # default to NOUN for content words


# ─────────────────────────────────────────────
#  7. NAMED ENTITY DETECTION (Pattern-Based)
# ─────────────────────────────────────────────

def detect_named_entities(sentence):
    """
    Detect potential named entities using capitalization patterns.

    Algorithm:
      - A Named Entity is a sequence of capitalized words NOT at the start of a sentence
      - Look for Title Case words in the middle of a sentence
      - Ignore first word (always capitalized)

    Returns a list of detected entity strings.
    """
    words = sentence.split()
    entities = []
    current_entity = []

    for i, word in enumerate(words):
        clean_word = word.strip(string.punctuation)
        # Skip first word (always capitalized), skip short words, skip numbers
        if i > 0 and clean_word and clean_word[0].isupper() and len(clean_word) > 2:
            current_entity.append(clean_word)
        else:
            if current_entity:
                entities.append(' '.join(current_entity))
                current_entity = []

    if current_entity:
        entities.append(' '.join(current_entity))

    return entities
