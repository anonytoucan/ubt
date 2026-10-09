"""S4b / math_eval-ko: Korean math braille (Korean Braille Standard 2024, mathematics).

The engine is the optional package ubt.kmath (latex_to_braille, sentence_to_braille, KMathError). If it does not
import, the stratum is reported `pending` with 0 rows; no other engine stands in.

f_kmath(latex)    = kmath.latex_to_braille(latex, strict=True)        standalone rows
f_kmath(sentence) = kmath.sentence_to_braille(sentence, strict=True)  Korean prose with $…$

Engine contract:
  * KMATH_TABLE_DIR / KMATH_TABLE_NAME point at the u1 engine's staged ko-2024 table (ubt.u1.engine.apply_env), so
    Korean prose in S4b and the S1/S7 Korean rows use the same table file;
  * output is 6-dot cells only (U+2800–U+283F), no space or newline; anything else is rejected as `charset`;
  * strict=True: derived [D] constructs raise KMathError(E_DERIVED) and are dropped and counted;
  * formulas come from a Korean school-curriculum grammar (ko_school_formula) because ML-paper formulas are mostly
    E_UNSUPPORTED; prose frames never put Latin letters in prose (E_PROSE_ROMAN_EDGE).

The regulation's own examples (resources/korean_braille_2024/examples_math.jsonl) certify the engine and are
registered eval-only (ubt.u1.registry).
"""
from __future__ import annotations

import random

KMATH_CELL_LO, KMATH_CELL_HI = 0x2800, 0x283F   # 6-dot only

_KO_PRE = [
    "다음 식을 계산하시오:", "다음 식을 간단히 하시오:", "방정식", "함수", "이때",
    "따라서", "위 조건에서", "다음이 성립함을 보이시오:", "학생이 쓴 답은", "주어진 식은",
    "그러므로", "예를 들어", "정의에 의하여", "양변을 정리하면", "여기서", "문제에서",
    "그래프에서", "조건", "등식", "부등식", "좌변은", "우변은",
]
_KO_POST = [
    "이다.", "을 만족한다.", "의 값을 구하시오.", "가 성립한다.", "이므로 참이다.",
    "을 풀어라.", "에서 최솟값을 가진다.", "로 나타낼 수 있다.", "이 된다.", "을 증명하시오.",
    "의 해를 구하여라.", "일 때를 생각하자.", "은 거짓이다.", "을 이용한다.", "와 같다.",
]


def load_kmath():
    """(module, None) or (None, reason)."""
    try:
        import ubt.kmath as km  # noqa: PLC0415
        for name in ("latex_to_braille", "sentence_to_braille", "KMathError"):
            if not hasattr(km, name):
                return None, f"ubt.kmath lacks {name}"
        return km, None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------- formulas
def ko_school_formula(rng: random.Random) -> tuple[str, str]:
    """(latex, family) from the Korean school curriculum, using only constructs the engine supports (no [D] items)."""
    d = lambda a=1, b=99: rng.randint(a, b)  # noqa: E731
    v = lambda: rng.choice("xyabnkmt")  # noqa: E731
    fam = rng.choice(["arith", "frac", "poly", "eq", "ineq", "root", "power", "func",
                      "log", "trig", "seq", "sum", "limit", "deriv", "integral", "set",
                      "prob", "abs"])
    x, y = v(), v()
    if fam == "arith":
        op = rng.choice(["+", "-", r"\times", r"\div"])
        f = f"{d()} {op} {d()} = {d(1, 999)}"
    elif fam == "frac":
        f = rng.choice([rf"\frac{{{d(1, 9)}}}{{{d(2, 12)}}}",
                        rf"\frac{{{d(1, 9)}}}{{{d(2, 9)}}} + \frac{{{d(1, 9)}}}{{{d(2, 9)}}}",
                        rf"\frac{{{x}+{d(1, 9)}}}{{{d(2, 9)}}}",
                        rf"\frac{{{d(1, 9)}}}{{{x}}}"])
    elif fam == "poly":
        f = rng.choice([rf"{x}^{{2}} + {d(1, 20)}{x} + {d(1, 30)}",
                        rf"({x}+{d(1, 9)})({x}-{d(1, 9)})",
                        rf"({x}+{y})^{{2}} = {x}^{{2}} + 2{x}{y} + {y}^{{2}}",
                        rf"{d(2, 9)}{x}^{{3}} - {d(1, 9)}{x}"])
    elif fam == "eq":
        f = rng.choice([rf"{d(2, 9)}{x} - {d(1, 20)} = {d(1, 40)}",
                        rf"{x}^{{2}} - {d(1, 9)}{x} + {d(1, 20)} = 0",
                        rf"{x} + {y} = {d(2, 20)}"])
    elif fam == "ineq":
        f = rng.choice([rf"{d(2, 9)}{x} + {d(1, 9)} > {d(1, 30)}",
                        rf"{x} \leq {d(1, 50)}", rf"{d(1, 9)} < {x} < {d(10, 20)}",
                        rf"{x} \geq -{d(1, 9)}"])
    elif fam == "root":
        f = rng.choice([rf"\sqrt{{{d(2, 99)}}}", rf"\sqrt{{{x}^{{2}}+{d(1, 9)}}}",
                        rf"{d(2, 9)}\sqrt{{{d(2, 7)}}}"])
    elif fam == "power":
        f = rng.choice([rf"{x}^{{{d(2, 9)}}}", rf"{d(2, 9)}^{{{d(2, 9)}}} = {d(4, 999)}",
                        rf"{x}^{{{d(2, 5)}}} \times {x}^{{{d(2, 5)}}}"])
    elif fam == "func":
        f = rng.choice([rf"f({x}) = {d(2, 9)}{x} + {d(1, 9)}",
                        rf"f({x}) = {x}^{{2}} - {d(1, 9)}",
                        rf"g({x}) = -{d(1, 9)}{x} + {d(1, 20)}"])
    elif fam == "log":
        base = rng.choice(["2", "3", "5", "10"])
        f = rng.choice([rf"\log_{{{base}}} {d(2, 99)}", rf"\log {d(2, 999)}",
                        rf"\log_{{{base}}} {x}"])
    elif fam == "trig":
        fn = rng.choice([r"\sin", r"\cos", r"\tan"])
        f = rng.choice([rf"{fn} {x}", rf"\sin^{{2}} {x} + \cos^{{2}} {x} = 1",
                        rf"{fn} {d(1, 89)}^{{\circ}}"])
    elif fam == "seq":
        f = rng.choice([rf"a_{{n}} = {d(2, 9)}n + {d(1, 9)}", rf"a_{{{d(1, 9)}}} = {d(1, 99)}",
                        rf"a_{{n+1}} = a_{{n}} + {d(1, 9)}"])
    elif fam == "sum":
        f = rf"\sum_{{k=1}}^{{{rng.choice(['n', str(d(2, 20))])}}} k"
    elif fam == "limit":
        f = rng.choice([rf"\lim_{{{x} \to {d(0, 9)}}} ({x}+{d(1, 9)})",
                        r"\lim_{n \to \infty} \frac{1}{n} = 0"])
    elif fam == "deriv":
        f = rng.choice([rf"f'({x}) = {d(2, 9)}{x}", rf"f'({x}) = {d(2, 9)}{x}^{{2}} + {d(1, 9)}"])
    elif fam == "integral":
        f = rng.choice([rf"\int_{{0}}^{{{d(1, 9)}}} {x} \, d{x}",
                        rf"\int {x}^{{{d(2, 5)}}} \, d{x}"])
    elif fam == "set":
        A, B = rng.sample("ABCPQ", 2)
        f = rng.choice([rf"{A} \cup {B}", rf"{A} \cap {B}", rf"{x} \in {A}",
                        rf"{A} \subset {B}", rf"n({A}) = {d(1, 20)}"])
    elif fam == "prob":
        A = rng.choice("ABE")
        f = rng.choice([rf"P({A}) = \frac{{{d(1, 5)}}}{{{d(6, 12)}}}",
                        rf"P({A}) + P({A}^{{c}}) = 1"])
    else:
        f = rng.choice([rf"|{x}| = {d(1, 9)}", rf"|{x} - {d(1, 9)}| < {d(1, 9)}"])
    return f, f"ko_school:{fam}"


def sample_ko_formula(rng: random.Random) -> str:
    return ko_school_formula(rng)[0]


def ko_sentence(rng: random.Random, latex: str) -> str:
    return f"{rng.choice(_KO_PRE)} ${latex}$ {rng.choice(_KO_POST)}"


def charset_ok(b: str) -> bool:
    return bool(b) and all(KMATH_CELL_LO <= ord(c) <= KMATH_CELL_HI for c in b)


def kmath_job(item: tuple[str, str]) -> tuple[str | None, str]:
    """Worker: (kind, text) -> (braille | None, reason), kind latex or sentence; needs KMATH_TABLE_DIR/NAME set."""
    km, why = load_kmath()
    if km is None:
        return None, "pending:" + why
    kind, text = item
    try:
        b = km.latex_to_braille(text, strict=True) if kind == "latex" \
            else km.sentence_to_braille(text, strict=True)
    except km.KMathError as e:
        return None, f"kmath_error:{getattr(e, 'code', type(e).__name__)}"
    except Exception as e:  # noqa: BLE001
        return None, f"exception:{type(e).__name__}"
    if not isinstance(b, str) or not b:
        return None, "empty"
    if not charset_ok(b):
        return None, "charset"
    return b, ""
