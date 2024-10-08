import json

_DEFAULT = 0
_FORWARD = 1
_IN_QUOTE = 2
_ESCAPE = 3
_IN_BLOCK = 4
_IN_LINE = 5
_OUT_BLOCK = 6


def _strip_comments(s):
    state = _DEFAULT
    t = []
    for c in s:
        if state == _DEFAULT:
            if c == '"':
                state = _IN_QUOTE
            elif c == '/':
                state = _FORWARD
                continue
            elif c == ' ' or c == '\r' or c == '\n':
                continue
        elif state == _IN_QUOTE:
            if c == '\\':
                state = _ESCAPE
            elif c == '"':
                state = _DEFAULT
        elif state == _FORWARD:
            if c == '/':
                state = _IN_LINE
                continue
            elif c == '*':
                state = _IN_BLOCK
                continue
        elif state == _ESCAPE:
            state = _IN_QUOTE
        elif state == _IN_LINE:
            if c == '\r' or c == '\n':
                state = _DEFAULT
            continue
        elif state == _IN_BLOCK:
            if c == '*':
                state = _OUT_BLOCK
            continue
        elif state == _OUT_BLOCK:
            if c == '/':
                state = _DEFAULT
            continue
        t.append(c)
    return ''.join(t)


def load(fp):
    s = _strip_comments(fp.read())
    return json.loads(s)
