import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ubt.math_norm import nu_math

def test_over_to_frac():
    assert nu_math(r"{1 \over 2}") == nu_math(r"\frac{1}{2}") == r"\frac{1}{2}"

def test_nested_over():
    assert nu_math(r"{a \over {b \over c}}") == r"\frac{a}{\frac{b}{c}}"

def test_script_bracing():
    assert nu_math(r"x^2") == "x^{2}"
    assert nu_math(r"a_\alpha") == r"a_{\alpha}"

def test_command_names_never_split():
    assert nu_math(r"\frac{1}{2}") == r"\frac{1}{2}"
    assert nu_math(r"\sum x") == r"\sum x"   # space kept: \sumx would differ

def test_aliases():
    assert nu_math(r"x \le y") == nu_math(r"x \leq y")
    assert nu_math(r"\dfrac{a}{b}") == nu_math(r"\frac{a}{b}")

def test_idempotent():
    s = nu_math(r"\sum_{i=1}^n {a_i \over 2}")
    assert nu_math(s) == s
