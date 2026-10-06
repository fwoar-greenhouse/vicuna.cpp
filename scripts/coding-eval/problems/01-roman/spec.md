Write a Roman numeral library.

- `toRoman(n)`: convert an integer from 1 to 3999 to its canonical Roman numeral (uppercase, subtractive
  forms IV, IX, XL, XC, CD, CM). Throw a `RangeError` for anything else (0, negative, > 3999, non-integers).
- `fromRoman(s)`: convert a Roman numeral string to a number. Accept upper or lower case. Accept only
  canonical numerals (the form `toRoman` produces); throw an `Error` for empty strings, invalid characters
  and non-canonical forms such as "IIII", "VV", "IC" or "XM".
- `calc(a, op, b)`: `a` and `b` are Roman numeral strings, `op` is one of "+", "-", "*", "/". Return the
  result as a canonical Roman numeral string. Throw an `Error` if an operand is invalid, if `op` is not one
  of the four operators, if division is not exact, or if the result is outside 1..3999.
