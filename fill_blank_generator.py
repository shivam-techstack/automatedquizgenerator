"""
fill_blank_generator.py
-----------------------
Fill in the Blanks Question Generator Module.

Algorithm:
  1. Extract sentences from the text
  2. For each sentence, identify the most important keyword
     using frequency score (higher frequency = more important)
  3. Replace that keyword with a blank: _______
  4. Return the sentence as a fill-in-the-blank question
  5. Also provide the correct answer (the removed word)

No pretrained models — uses frequency-based keyword importance.
"""

import re
import random
from text_processing import (
    tokenize_sentences,
    tokenize_words,
    remove_stopwords,
    extract_keywords_using_frequency
)
from collections import Counter


def generate_fill_blanks(text, num_questions=5):
    """
    Generate Fill-in-the-Blank questions from the given text.

    Parameters:
      text          : cleaned input academic text
      num_questions : number of fill-blank questions to generate

    Returns:
      List of dicts, each with:
        - question : sentence with one word replaced by _______
        - answer   : the word that was removed
        - hint     : first letter of the answer as a hint
    """
    sentences = tokenize_sentences(text)
    global_keywords = extract_keywords_using_frequency(text, top_n=25)

    if not sentences or not global_keywords:
        return []

    questions = []
    used_sentences = set()

    # Build frequency map for scoring within each sentence
    all_words = tokenize_words(text)
    filtered_all = remove_stopwords(all_words)
    freq_map = Counter(filtered_all)

    for sentence in sentences:
        if len(questions) >= num_questions:
            break

        if sentence in used_sentences:
            continue

        # Get content words in this sentence
        words = tokenize_words(sentence)
        content_words = remove_stopwords(words)

        if not content_words:
            continue

        # ── Select the best blank word ──
        # Score each content word by its global frequency
        # Higher frequency = more important = better blank
        scored = [(w, freq_map.get(w, 0)) for w in content_words if len(w) > 3]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Pick the top-scored word as the blank
        blank_word = None
        for word, score in scored:
            if word in global_keywords:
                blank_word = word
                break

        # Fallback: just take the highest-frequency content word
        if not blank_word and scored:
            blank_word = scored[0][0]

        if not blank_word:
            continue

        # ── Create the fill-in-the-blank question ──
        question_text = re.sub(
            r'\b' + re.escape(blank_word) + r'\b',
            '_______',
            sentence,
            count=1,  # Only replace first occurrence
            flags=re.IGNORECASE
        )

        if '_______' not in question_text:
            continue

        used_sentences.add(sentence)

        # Generate a hint: first letter + length
        hint = f"{blank_word[0].upper()}{'_' * (len(blank_word) - 1)} ({len(blank_word)} letters)"

        questions.append({
            "question": question_text,
            "answer": blank_word.capitalize(),
            "hint": hint
        })

    return questions
