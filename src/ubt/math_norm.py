"""ν_math: canonical LaTeX normalization.

f_math = latex2nemeth ∘ ν_math must depend on the formula, not on how it was typed: `\frac{1}{2}` and `{1 \over 2}`
must reach the converter as one string, as NFC does for text (§3.1); otherwise notation variants inflate the fiber.

Scope: comments stripped; {A \over B} -> \frac{A}{B}; \dfrac/\tfrac -> \frac; relation aliases (\le \ge \ne \to);
single-token scripts and \sqrt arguments braced; whitespace removed except where a \command would swallow a letter.
"""

from __future__ import annotations

import re

_ALIASES = {
    r"\dfrac": r"\frac", r"\tfrac": r"\frac",
    r"\le": r"\leq", r"\ge": r"\geq", r"\ne": r"\neq",
    r"\to": r"\rightarrow",
}
# longest first so \leq is not hit by the \le rule
_ALIAS_RE = re.compile(
    "|".join(re.escape(k) + r"(?![a-zA-Z])" for k in
             sorted(_ALIASES, key=len, reverse=True)))


def _find_group_start(s: str, close: int) -> int:
    """index of the '{' matching s[close]=='}' scanning backwards."""
    depth = 0
    for i in range(close, -1, -1):
        if s[i] == "}":
            depth += 1
        elif s[i] == "{":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _over_to_frac(s: str) -> str:
    """{A \over B} -> \frac{A}{B}, innermost-first."""
    while True:
        m = re.search(r"\\over(?![a-zA-Z])", s)
        if not m:
            return s
        # numerator: back to enclosing '{'
        i = m.start() - 1
        depth = 0
        while i >= 0:
            if s[i] == "}":
                depth += 1
            elif s[i] == "{":
                if depth == 0:
                    break
                depth -= 1
            i -= 1
        if i < 0:
            return s  # malformed; leave as-is
        # denominator: forward to enclosing '}'
        j = m.end()
        depth = 0
        while j < len(s):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                if depth == 0:
                    break
                depth -= 1
            j += 1
        if j >= len(s):
            return s
        num = s[i + 1:m.start()].strip()
        den = s[m.end():j].strip()
        s = s[:i] + "\\frac{" + num + "}{" + den + "}" + s[j + 1:]


def nu_math(s: str) -> str:
    s = re.sub(r"(?<!\\)%.*", "", s)              # comments
    s = _over_to_frac(s)
    s = _ALIAS_RE.sub(lambda m: _ALIASES[m.group(0)], s)
    # brace single-token scripts and \sqrt args
    s = re.sub(r"([_^])\s*(\\[a-zA-Z]+|[0-9a-zA-Z])(?!\s*[{a-zA-Z0-9])",
               r"\1{\2}", s)
    s = re.sub(r"([_^])\s*([0-9a-zA-Z])(?=[^{])", r"\1{\2}", s)
    s = re.sub(r"(\\sqrt)\s*([0-9a-zA-Z])", r"\1{\2}", s)
    # drop whitespace but keep one space between a \command and a following letter (\sum x, not \sumx);
    # marking first stops the command regex from backtracking into a name (\frac, not \fra c).
    s = re.sub(r"\s+", "\x00", s)
    s = re.sub(r"(\\[a-zA-Z]+)\x00(?=[a-zA-Z])", r"\1 ", s)
    s = s.replace("\x00", "")
    return s
