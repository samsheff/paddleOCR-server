"""Dump the raw generated token stream for one page.

Used to tune the repetition guard in app/ocr.py: point it at a page that
repeats, and it reports the offset at which output restarts from the top.

    python scripts/inspect_tokens.py samples/sample.pdf 2 1605632

Arguments: PDF path, 1-based page, max_pixels.
"""
