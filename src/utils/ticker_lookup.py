# src/utils/ticker_lookup.py
import builtins

def find_ticker_interactive(name_or_ticker):
    """
    Try to be flexible:
    - If input looks like a ticker (contains '.' or all uppercase + digits), return it.
    - Otherwise, ask user to provide exact ticker (fallback).
    """
    s = name_or_ticker.strip()
    # basic heuristic: if contains dot like TCS.NS or contains uppercase and maybe digits
    if "." in s or s.isupper() or any(ch.isdigit() for ch in s):
        return s
    # Not a ticker — ask user for ticker
    print("Could not auto-resolve. Please provide the ticker (e.g. TCS.NS) for:", s)
    t = input("Enter ticker: ").strip()
    return t
