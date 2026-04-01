"""
mcq_generator.py
----------------
Multiple Choice Question (MCQ) Generator Module.

Algorithm:
  1. Extract keywords from the text (frequency-based)
  2. For each keyword, find the sentence it appears in
  3. Replace the keyword with a blank → that becomes the question stem
  4. Generate 3 distractor options using:
       - Other keywords of similar POS from the same text
       - Random selection to ensure variety
  5. Shuffle options so correct answer is not always in same position
  6. Return structured MCQ data

No pretrained models used — pure rule-based logic.
"""

import random
import re
from text_processing import (
    tokenize_sentences,
    extract_keywords_using_frequency,
    remove_stopwords,
    tokenize_words,
    simple_pos_tag
)


def generate_mcqs(text, num_questions=5):
    """
    Generate Multiple Choice Questions from the given text.

    Parameters:
      text          : cleaned input academic text
      num_questions : number of MCQs to generate (default 5)

    Returns:
      List of dicts, each with:
        - question   : question string with blank
        - options    : list of 4 option strings ["A. x", "B. y", ...]
        - answer     : correct answer string
    """
    sentences = tokenize_sentences(text)
    keywords = extract_keywords_using_frequency(text, top_n=20)

    if not keywords or not sentences:
        return []

    mcqs = []
    used_sentences = set()

    for keyword in keywords:
        if len(mcqs) >= num_questions:
            break

        # Find a sentence that contains this keyword (case-insensitive)
        target_sentence = None
        for sent in sentences:
            if keyword.lower() in sent.lower() and sent not in used_sentences:
                target_sentence = sent
                break

        if not target_sentence:
            continue

        used_sentences.add(target_sentence)

        # ── Create question stem by replacing keyword with blank ──
        # Use regex for case-insensitive replacement
        question_stem = re.sub(
            r'\b' + re.escape(keyword) + r'\b',
            '_______',
            target_sentence,
            flags=re.IGNORECASE
        )

        # Skip if the blank was not inserted (keyword wasn't a whole word)
        if '_______' not in question_stem:
            continue

        # ── Generate distractors ──
        distractors = _generate_distractors(keyword, keywords, text, num_distractors=3)

        if len(distractors) < 3:
            continue  # Skip if we can't make 4 options total

        # ── Build options list and shuffle ──
        options = [keyword] + distractors[:3]
        random.shuffle(options)

        labeled_options = []
        for i, opt in enumerate(options):
            label = chr(65 + i)  # A, B, C, D
            labeled_options.append(f"{label}. {opt.capitalize()}")

        # Find which label is the correct answer
        correct_label = None
        for i, opt in enumerate(options):
            if opt.lower() == keyword.lower():
                correct_label = chr(65 + i)
                break

        mcqs.append({
            "question": question_stem,
            "options": labeled_options,
            "answer": f"{correct_label}. {keyword.capitalize()}"
        })

    return mcqs


def _generate_distractors(keyword, all_keywords, text, num_distractors=3):
    """
    Generate distractor options for an MCQ.

    Distractor Generation Algorithm:
      1. Get POS tag of the correct keyword
      2. From all_keywords, find words with the SAME POS tag (similar category)
      3. Exclude the correct keyword itself
      4. If not enough same-POS words, fall back to any other keywords
      5. Return up to num_distractors words

    This ensures distractors are plausible (same grammatical category).
    """
    target_pos = simple_pos_tag(keyword)

    # Filter keywords: same POS, not the keyword itself, minimum length
    same_pos = [
        k for k in all_keywords
        if k.lower() != keyword.lower()
        and simple_pos_tag(k) == target_pos
        and len(k) > 2
    ]

    # Fallback: any keyword that's not the target
    other_keywords = [
        k for k in all_keywords
        if k.lower() != keyword.lower() and len(k) > 2
    ]

    # Prefer same-POS distractors, then fill with others
    distractors = same_pos[:num_distractors]

    if len(distractors) < num_distractors:
        remaining = [k for k in other_keywords if k not in distractors]
        random.shuffle(remaining)
        distractors += remaining[:num_distractors - len(distractors)]

    random.shuffle(distractors)
    return distractors[:num_distractors]
