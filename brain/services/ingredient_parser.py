from __future__ import annotations

import re
from typing import List, Optional, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, Field


router = APIRouter(prefix="/ingredients", tags=["Ingredients"])


# ----------------------------
# Models
# ----------------------------

class ParseIngredientsRequest(BaseModel):
    lines: List[str] = Field(..., description="Raw ingredient lines as written in recipe")


class Quantity(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    raw: Optional[str] = None  # original quantity text e.g. "1 1/2", "1-2", "to taste"


class ParsedIngredient(BaseModel):
    original: str
    qty: Quantity = Field(default_factory=Quantity)
    unit: Optional[str] = None
    name: Optional[str] = None
    preparation: Optional[str] = None  # e.g. "diced", "minced"
    notes: List[str] = Field(default_factory=list)
    confidence: str = Field(..., description="high | medium | low")


class ParseIngredientsResponse(BaseModel):
    parsed: List[ParsedIngredient]


# ----------------------------
# Parsing helpers
# ----------------------------

UNIT_ALIASES = {
    # volume
    "tsp": "teaspoon",
    "tsps": "teaspoon",
    "teaspoon": "teaspoon",
    "teaspoons": "teaspoon",
    "tbsp": "tablespoon",
    "tbsps": "tablespoon",
    "tablespoon": "tablespoon",
    "tablespoons": "tablespoon",
    "cup": "cup",
    "cups": "cup",
    "pt": "pint",
    "pint": "pint",
    "pints": "pint",
    "qt": "quart",
    "quart": "quart",
    "quarts": "quart",
    "gal": "gallon",
    "gallon": "gallon",
    "gallons": "gallon",
    "ml": "ml",
    "l": "l",
    "liter": "l",
    "liters": "l",
    # weight
    "oz": "oz",
    "ounce": "oz",
    "ounces": "oz",
    "lb": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "g": "g",
    "gram": "g",
    "grams": "g",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    # count-ish
    "clove": "clove",
    "cloves": "clove",
    "can": "can",
    "cans": "can",
    "jar": "jar",
    "jars": "jar",
    "package": "package",
    "packages": "package",
    "pkg": "package",
    "pkgs": "package",
    "slice": "slice",
    "slices": "slice",
    "stick": "stick",
    "sticks": "stick",
    "piece": "piece",
    "pieces": "piece",
}

# Common non-quantified markers
NON_QUANT = {
    "to taste",
    "as needed",
    "optional",
}

# Unicode fraction map (common ones)
UNICODE_FRACTIONS = {
    "¼": "1/4",
    "½": "1/2",
    "¾": "3/4",
    "⅓": "1/3",
    "⅔": "2/3",
    "⅛": "1/8",
    "⅜": "3/8",
    "⅝": "5/8",
    "⅞": "7/8",
}


def _normalize_text(s: str) -> str:
    s = s.strip()
    for u, ascii_frac in UNICODE_FRACTIONS.items():
        s = s.replace(u, ascii_frac)
    # normalize dash variants to hyphen
    s = s.replace("–", "-").replace("—", "-")
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _to_float(num: str) -> Optional[float]:
    """
    Convert strings like "1", "1/2", "1 1/2" to float.
    """
    num = num.strip()
    if not num:
        return None

    # mixed number: "1 1/2"
    if re.match(r"^\d+\s+\d+/\d+$", num):
        a, b = num.split()
        f = _to_float(b)
        if f is None:
            return None
        return float(a) + f

    # fraction: "1/2"
    if re.match(r"^\d+/\d+$", num):
        n, d = num.split("/")
        d_i = int(d)
        if d_i == 0:
            return None
        return int(n) / d_i

    # decimal/int
    if re.match(r"^\d+(\.\d+)?$", num):
        return float(num)

    return None


def _parse_quantity_prefix(s: str) -> Tuple[Quantity, str, List[str]]:
    """
    Parse quantity at the start of the string.
    Returns (Quantity, remainder, notes)
    Supports:
      - "1"
      - "1.5"
      - "1/2"
      - "1 1/2"
      - "1-2"
      - "1 to 2"
      - "to taste" / "as needed"
    """
    notes: List[str] = []
    s0 = s.strip().lower()

    # Non-quantified phrases at start
    for phrase in NON_QUANT:
        if s0.startswith(phrase):
            q = Quantity(min=None, max=None, raw=phrase)
            remainder = s[len(phrase):].strip(" ,")
            notes.append(f"Non-quantified quantity: '{phrase}'")
            return q, remainder, notes

    # Pattern: "1 to 2 ..."
    m = re.match(
        r"^(\d+(?:\.\d+)?|\d+\s+\d+/\d+|\d+/\d+)\s+to\s+(\d+(?:\.\d+)?|\d+\s+\d+/\d+|\d+/\d+)\b(.*)$",
        s,
        re.IGNORECASE,
    )
    if m:
        a_raw, b_raw, rest = m.group(1), m.group(2), m.group(3).strip()
        a = _to_float(a_raw)
        b = _to_float(b_raw)
        q = Quantity(min=a, max=b, raw=f"{a_raw} to {b_raw}")
        notes.append("Range quantity parsed using 'to'.")
        return q, rest, notes

    # Pattern: "1-2 ..." or "1 - 2 ..."
    m = re.match(r"^(\d+(?:\.\d+)?|\d+\s+\d+/\d+|\d+/\d+)\s*-\s*(\d+(?:\.\d+)?|\d+\s+\d+/\d+|\d+/\d+)\b(.*)$", s)
    if m:
        a_raw, b_raw, rest = m.group(1), m.group(2), m.group(3).strip()
        a = _to_float(a_raw)
        b = _to_float(b_raw)
        q = Quantity(min=a, max=b, raw=f"{a_raw}-{b_raw}")
        notes.append("Range quantity parsed using '-'.")
        return q, rest, notes

    # Single number/fraction/mixed at start
    # IMPORTANT: order matters. Prefer mixed and fractions before integers.
    m = re.match(r"^(\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)\b(.*)$", s)
    if m:
        a_raw, rest = m.group(1), m.group(2).strip()
        a = _to_float(a_raw)
        q = Quantity(min=a, max=a, raw=a_raw)
        return q, rest, notes

    # No quantity prefix
    return Quantity(min=None, max=None, raw=None), s, notes


def _parse_unit_prefix(s: str) -> Tuple[Optional[str], str]:
    """
    Parse a unit at the start of the remainder.
    """
    if not s:
        return None, s

    s = s.strip(" ,")

    m = re.match(r"^([A-Za-z]+)\b(.*)$", s)
    if not m:
        return None, s

    token = m.group(1).lower()
    rest = m.group(2).strip()

    if token in UNIT_ALIASES:
        return UNIT_ALIASES[token], rest

    return None, s


def _split_preparation(name_part: str) -> Tuple[str, Optional[str]]:
    """
    Try to detect preparation after comma, e.g. "onion, diced"
    """
    if "," in name_part:
        left, right = name_part.split(",", 1)
        prep = right.strip()
        if prep:
            return left.strip(), prep
    return name_part.strip(), None


def _extract_parenthetical_notes(s: str) -> Tuple[str, List[str]]:
    """
    Pull parenthetical segments into notes: "1 (28 ounce) jar ..."
    """
    notes: List[str] = []

    def repl(m: re.Match) -> str:
        content = m.group(1).strip()
        if content:
            notes.append(f"Parenthetical: {content}")
        return " "

    out = re.sub(r"\(([^)]+)\)", repl, s)
    out = re.sub(r"\s+", " ", out).strip()
    return out, notes


def parse_ingredient_line(line: str) -> ParsedIngredient:
    original = line
    notes: List[str] = []

    s = _normalize_text(line)
    s, paren_notes = _extract_parenthetical_notes(s)
    notes.extend(paren_notes)

    # quantity
    qty, remainder, qty_notes = _parse_quantity_prefix(s)
    notes.extend(qty_notes)

    # unit
    unit, remainder2 = _parse_unit_prefix(remainder)

    # name + preparation
    name_raw = remainder2.strip(" ,")
    name, prep = _split_preparation(name_raw)

    # Confidence scoring (simple + explainable)
    confidence = "high"

    if qty.raw is None and unit is None:
        confidence = "medium"

    if not name:
        confidence = "low"
        notes.append("Could not confidently extract ingredient name.")

    if qty.raw and qty.min is None and qty.max is None and qty.raw not in NON_QUANT:
        confidence = "low"
        notes.append(f"Quantity '{qty.raw}' could not be parsed into a number.")

    if qty.raw in NON_QUANT:
        confidence = "high"

    return ParsedIngredient(
        original=original,
        qty=qty,
        unit=unit,
        name=name if name else None,
        preparation=prep,
        notes=notes,
        confidence=confidence,
    )


# ----------------------------
# API
# ----------------------------

@router.post("/parse", response_model=ParseIngredientsResponse)
async def parse_ingredients(payload: ParseIngredientsRequest) -> ParseIngredientsResponse:
    parsed = [parse_ingredient_line(line) for line in payload.lines]
    return ParseIngredientsResponse(parsed=parsed)
