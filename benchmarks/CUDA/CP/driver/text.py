#! /usr/bin/env python

from itertools import imap
import re

TOKEN = re.compile(r"(?:^|(?<=\s))[^\s]+|(?:^|(?<!\s))[\s]+")

COLUMNS=80

class iscan:
    """A scan iterator."""
    def __init__(self, f, init, s):
        self.f = f
        self.s = s.__iter__()
        self.current = init
        self.begin = True

    def next(self):
        if self.begin:
            self.begin = False
            return self.current
        else:
            cur = self.f(self.current, self.s.next()) # raises StopIteration
            self.current = cur
            return cur

    def __iter__(self): return self

def format_columns(text, indent=0):
    """Format the text to fit in 80 columns by inserting newlines.  Spaces
    are inserted at the beginning of each line to indent by 'indent'
    columns."""

    outls = []
    for inl in text.split('\n'):
        # Split the line into alternating space and non-space tokens
        tokens = TOKEN.findall(inl)

        # Empty line?
        if not tokens:
            outls.append("")
            continue

        # Use the spaces at the beginning of the line to determine
        # the line's indentation
        if tokens[0].isspace():
            inl_indent = indent + len(tokens[0])
            del tokens[0]
        else:
            inl_indent = indent

        # Write tokens into the output
        while tokens:
            # Ignore whitespace at the beginning of the line
            if tokens[0].isspace(): del tokens[0]

            # Take as many tokens as possible without exceeding the line
            # length; however, must take at least one token
            n = 0
            lengths = iscan(lambda n, t: n + len(t), inl_indent, tokens)
            lengths.next()
            for c in lengths:
                if c > COLUMNS: break
                n += 1
            n = max(n, 1)

            # Concatenate the tokens and prepend spaces for indentation
            outls.append(' ' * inl_indent + "".join(tokens[:n]))

            # Continue with tokens
            tokens = tokens[n:]

    return "\n".join(outls)
            
    
