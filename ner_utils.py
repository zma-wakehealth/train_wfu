import numpy as np
import html

def normalize_text(text, replace_newlines=True):
    """
    Standardizes the text by unescaping HTML and normalizing whitespace.
    This MUST match the logic used during model inference.
    """
    if not text:
        return ""
    # Unescape healthy HTML (e.g., &apos; -> ')
    text = html.unescape(text)
    # Standardize whitespace
    text = text.replace('\r', ' ').replace('\t', ' ')
    if replace_newlines:
        text = text.replace('\n', ' ')
    return text
