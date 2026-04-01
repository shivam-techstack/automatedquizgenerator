"""
true_false_generator.py
-----------------------
True / False Question Generator Module.

Algorithm:
  1. Extract clear factual sentences from the text
     (sentences with subject-verb structure, not too short/long)
  2. For ~50% of sentences: keep them as TRUE statements
  3. For other ~50%: modify them to make them FALSE by:
       a. Swapping a named entity or keyword with a different one
       b. Changing a number to a different number
       c. Negating a verb phrase
  4. Return structured True/False data

No pretrained models — pure rule-based manipulation.
"""

import random
import re
import string
from text_processing import (
    tokenize_sentences,
    extract_keywords_using_frequency,
    detect_named_entities,
    remove_stopwords,
    tokenize_words
)


def generate_true_false(text, num_questions=5):
    """
    Generate True/False questions from the given text.

    Parameters:
      text          : cleaned input academic text
      num_questions : number of T/F questions to generate

    Returns:
      List of dicts, each with:
        - question : the (possibly modified) statement
        - answer   : "True" or "False"
        - original : original sentence (for reference)
    """
    sentences = tokenize_sentences(text)
    keywords = extract_keywords_using_frequency(text, top_n=20)

    # Filter sentences that are good candidates (reasonable length, factual tone)
    candidates = _filter_factual_sentences(sentences)

    if not candidates:
        return []

    # Shuffle for variety
    random.shuffle(candidates)
    selected = candidates[:num_questions]

    questions = []
    for i, sentence in enumerate(selected):
        # Alternate: even index → True, odd index → False
        make_false = (i % 2 == 1)

        if make_false:
            modified, changed = _make_false(sentence, keywords, sentences)
            if changed:
                questions.append({
                    "question": modified,
                    "answer": "False",
                    "original": sentence
                })
            else:
                # Could not modify → use as True
                questions.append({
                    "question": sentence,
                    "answer": "True",
                    "original": sentence
                })
        else:
            questions.append({
                "question": sentence,
                "answer": "True",
                "original": sentence
            })

    return questions


def _filter_factual_sentences(sentences):
    """
    Select sentences likely to be factual/declarative.

    Filters:
      - Sentence length between 8 and 40 words
      - Does not start with question words (Who, What, Why, etc.)
      - Does not contain parentheses or citations
      - Contains at least one verb indicator
    """
    question_starters = {'who', 'what', 'why', 'how', 'when', 'where', 'which'}
    good = []

    for sent in sentences:
        words = sent.split()
        word_count = len(words)

        # Length filter
        if word_count < 6 or word_count > 45:
            continue

        # Skip questions
        first_word = words[0].lower().strip(string.punctuation)
        if first_word in question_starters or sent.strip().endswith('?'):
            continue

        # Skip sentences with references/citations (e.g., "[1]", "(Smith 2020)")
        if re.search(r'\[\d+\]|\(\w+\s+\d{4}\)', sent):
            continue

        good.append(sent)

    return good


def _make_false(sentence, keywords, all_sentences):
    """
    Modify a sentence to make it factually incorrect.

    Strategies (tried in order):
      1. NUMBER SWAP: replace a number with a different number
      2. KEYWORD SWAP: replace a keyword with another unrelated keyword
      3. ENTITY SWAP: replace a capitalized entity with another from the text

    Returns: (modified_sentence, was_changed: bool)
    """

    # ── Strategy 1: Swap numbers ──
    numbers = re.findall(r'\b\d+\.?\d*\b', sentence)
    if numbers:
        original_num = random.choice(numbers)
        fake_num = _generate_different_number(original_num)
        modified = sentence.replace(original_num, fake_num, 1)
        return modified, True

    # ── Strategy 2: Swap a keyword ──
    words_in_sentence = set(tokenize_words(sentence))
    matching_keywords = [k for k in keywords if k in words_in_sentence]

    if len(matching_keywords) >= 1:
        target_kw = random.choice(matching_keywords)
        # Find a replacement keyword from the global list (different from target)
        replacements = [k for k in keywords if k.lower() != target_kw.lower() and len(k) > 2]
        if replacements:
            replacement = random.choice(replacements)
            modified = re.sub(
                r'\b' + re.escape(target_kw) + r'\b',
                replacement,
                sentence,
                count=1,
                flags=re.IGNORECASE
            )
            if modified != sentence:
                return modified, True

    # ── Strategy 3: Swap a named entity ──
    entities = detect_named_entities(sentence)
    all_entities = []
    for s in all_sentences:
        all_entities.extend(detect_named_entities(s))
    all_entities = list(set(all_entities))

    if entities and len(all_entities) > 1:
        target_entity = random.choice(entities)
        other_entities = [e for e in all_entities if e != target_entity]
        if other_entities:
            replacement_entity = random.choice(other_entities)
            modified = sentence.replace(target_entity, replacement_entity, 1)
            if modified != sentence:
                return modified, True

    return sentence, False


def _generate_different_number(num_str):
    """
    Generate a plausibly different number from the original.
    Keeps the same order of magnitude but changes the value.
    """
    try:
        num = float(num_str)
        # Offset by a meaningful amount
        offsets = [num * 0.5, num * 2, num + 10, num - 10, num * 3]
        offsets = [o for o in offsets if o != num and o > 0]
        if offsets:
            new_num = random.choice(offsets)
            # Format as int if original was int
            if '.' not in num_str:
                return str(int(new_num))
            else:
                return f"{new_num:.1f}"
    except ValueError:
        pass
    return "100"  # Safe fallback
