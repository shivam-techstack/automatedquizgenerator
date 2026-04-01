"""
question_generator.py
---------------------
Central Controller / Orchestrator for the Automated Quiz Generator.

This module ties together all sub-generators:
  - MCQ Generator
  - True/False Generator
  - Fill in the Blanks Generator

It handles:
  1. Input text preprocessing
  2. Calling each generator
  3. Combining results into a single response dict
"""

from text_processing import clean_text
from mcq_generator import generate_mcqs
from true_false_generator import generate_true_false
from fill_blank_generator import generate_fill_blanks


def generate_quiz(raw_text, num_mcq=5, num_tf=5, num_fill=5):
    """
    Main entry point for quiz generation.

    Pipeline:
      raw_text
        → clean_text()           [text_processing.py]
        → generate_mcqs()        [mcq_generator.py]
        → generate_true_false()  [true_false_generator.py]
        → generate_fill_blanks() [fill_blank_generator.py]
        → combined JSON response

    Parameters:
      raw_text  : The academic text input from the user
      num_mcq   : Number of MCQ questions to generate
      num_tf    : Number of True/False questions to generate
      num_fill  : Number of Fill-in-the-blank questions to generate

    Returns:
      dict with keys: 'mcq', 'true_false', 'fill_blanks'
    """

    # ── Step 1: Clean and validate input ──
    if not raw_text or not raw_text.strip():
        return {
            "error": "Input text is empty. Please provide academic text.",
            "mcq": [],
            "true_false": [],
            "fill_blanks": []
        }

    text = clean_text(raw_text)

    if len(text.split()) < 30:
        return {
            "error": "Text is too short. Please provide at least 30 words.",
            "mcq": [],
            "true_false": [],
            "fill_blanks": []
        }

    # ── Step 2: Generate all question types ──
    try:
        mcqs = generate_mcqs(text, num_questions=num_mcq)
    except Exception as e:
        mcqs = []
        print(f"[MCQ Error] {e}")

    try:
        tf_questions = generate_true_false(text, num_questions=num_tf)
    except Exception as e:
        tf_questions = []
        print(f"[T/F Error] {e}")

    try:
        fill_questions = generate_fill_blanks(text, num_questions=num_fill)
    except Exception as e:
        fill_questions = []
        print(f"[Fill Error] {e}")

    # ── Step 3: Return combined result ──
    return {
        "mcq": mcqs,
        "true_false": tf_questions,
        "fill_blanks": fill_questions
    }
