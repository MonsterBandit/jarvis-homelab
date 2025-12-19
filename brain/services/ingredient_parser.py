from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

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
    preparation: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
    confidence: str = Field(..., description="high | medium | low")


class ParseIngredientsResponse(BaseModel):
    parsed: List[ParsedIngredient]


# ----------------------------
# Parsing helpers
# ----------------------------

UNIT_ALIASES = {
    "tsp": "teaspoon", "tsps": "teaspoon", "teaspoon": "teaspoon", "teaspoons": "teaspoon",
    "tbsp": "tablespoon", "tbsps": "tablespoon", "tablespoon": "tablespoon", "tablespoons": "tablespoon",
    "cup": "cup", "cups": "cup",
    "pt": "pint", "pint": "pint", "pints": "pint",
    "qt": "quart", "quart": "quart", "quarts": "quart",
    "gal": "gallon", "gallon": "gallon", "gallons": "gallon",
    "ml": "ml", "l": "l", "liter": "l", "liters": "l",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "g": "g", "gram": "g", "grams": "g",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "clove": "clove", "cloves": "clove",
    "can": "can", "cans": "can",
    "jar": "jar", "jars": "jar",
    "package": "package", "packages": "package", "pkg": "package", "pkgs": "package",
    "slice": "slice", "slices": "slice",
    "stick": "stick", "sticks": "stick",
    "piece": "piece", "pieces": "piece",
}

NON_QUANT = {"to taste", "as needed", "optional"}

UNICODE_FRACTIONS = {
    "¼": "1/4", "½": "1/2", "¾": "3/4",
    "⅓": "1/3", "⅔": "2/3",
    "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
}


def _normalize_text(s: str) -> str:
    s = s.strip()
    for u, ascii_frac in UNICODE_FRACTIONS.items():
        s = s.replace(u, ascii_frac)
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def _to_float(num: str) -> Optional[float]:
    num = num.strip()
    if not num:
        return None

    if re.match(r"^\d+\s+\d+/\d+$", num):
        a, b = num.split()
        f = _to_float(b)
        return float(a) + f if f is not None else None

    if re.match(r"^\d+/\d+$", num):
        n, d = num.split("/")
        d_i = int(d)
        return int(n) / d_i if d_i != 0 else None

    if re.match(r"^\d+(\.\d+)?$", num):
        return float(num)

    return None


def _parse_quantity_prefix(s: str) -> Tuple[Quantity, str, List[str]]:
    notes: List[str] = []
    s0 = s.strip().lower()

    for phrase in NON_QUANT:
        if s0.startswith(phrase):
            q = Quantity(min=None, max=None, raw=phrase)
            remainder = s[len(phrase):].strip(" ,")
            notes.append(f"Non-quantified quantity: '{phrase}'")
            return q, remainder, notes

    m = re.match(
        r"^(\d+(?:\.\d+)?|\d+\s+\d+/\d+|\d+/\d+)\s+to\s+(\d+(?:\.\d+)?|\d+\s+\d+/\d+|\d+/\d+)\b(.*)$",
        s, re.IGNORECASE,
    )
    if m:
        a_raw, b_raw, rest = m.group(1), m.group(2), m.group(3).strip()
        q = Quantity(min=_to_float(a_raw), max=_to_float(b_raw), raw=f"{a_raw} to {b_raw}")
        notes.append("Range quantity parsed using 'to'.")
        return q, rest, notes

    m = re.match(
        r"^(\d+(?:\.\d+)?|\d+\s+\d+/\d+|\d+/\d+)\s*-\s*(\d+(?:\.\d+)?|\d+\s+\d+/\d+|\d+/\d+)\b(.*)$",
        s,
    )
    if m:
        a_raw, b_raw, rest = m.group(1), m.group(2), m.group(3).strip()
        q = Quantity(min=_to_float(a_raw), max=_to_float(b_raw), raw=f"{a_raw}-{b_raw}")
        notes.append("Range quantity parsed using '-'.")
        return q, rest, notes

    m = re.match(r"^(\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)\b(.*)$", s)
    if m:
        a_raw, rest = m.group(1), m.group(2).strip()
        q = Quantity(min=_to_float(a_raw), max=_to_float(a_raw), raw=a_raw)
        return q, rest, notes

    return Quantity(min=None, max=None, raw=None), s, notes


def _parse_unit_prefix(s: str) -> Tuple[Optional[str], str]:
    if not s:
        return None, s
    s = s.strip(" ,")
    m = re.match(r"^([A-Za-z]+)\b(.*)$", s)
    if not m:
        return None, s
    token = m.group(1).lower()
    rest = m.group(2).strip()
    return (UNIT_ALIASES[token], rest) if token in UNIT_ALIASES else (None, s)


def _split_preparation(name_part: str) -> Tuple[str, Optional[str]]:
    if "," in name_part:
        left, right = name_part.split(",", 1)
        prep = right.strip()
        if prep:
            return left.strip(), prep
    return name_part.strip(), None


def _extract_parenthetical_notes(s: str) -> Tuple[str, List[str]]:
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

    qty, remainder, qty_notes = _parse_quantity_prefix(s)
    notes.extend(qty_notes)

    unit, remainder2 = _parse_unit_prefix(remainder)
    name_raw = remainder2.strip(" ,")
    name, prep = _split_preparation(name_raw)

    confidence = "high"
    if qty.raw is None and unit is None:
        confidence = "medium"
    if not name:
        confidence = "low"
        notes.append("Could not confidently extract ingredient name.")

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
# PUBLIC HELPER (Phase 6.75 fix)
# ----------------------------

def parse_ingredient_lines(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Canonical parsing helper.
    Used by:
      - POST /ingredients/parse
      - POST /recipes/draft
    """
    return [parse_ingredient_line(line).model_dump() for line in lines]


# ----------------------------
# API
# ----------------------------

@router.post("/parse", response_model=ParseIngredientsResponse)
async def parse_ingredients(payload: ParseIngredientsRequest) -> ParseIngredientsResponse:
    parsed = [parse_ingredient_line(line) for line in payload.lines]
    return ParseIngredientsResponse(parsed=parsed)
