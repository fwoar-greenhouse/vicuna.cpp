Write money helpers that work in integer cents (never use floating point for amounts).

- `parseMoney(s)`: parse a US dollar amount and return integer cents. Accepted forms: optional `-` sign,
  optional `$`, digits with optional `,` thousands separators (if commas are used they must be every three
  digits), optional `.` followed by exactly two digits. Accounting negatives in parentheses are also
  accepted: `($1,234.50)`. Examples: `"$1,234.56"` -> 123456, `"-$0.05"` -> -5, `"12"` -> 1200,
  `"(3.10)"` -> -310. Throw a `SyntaxError` for anything else (`"1.5"`, `"1,23"`, `"$-1"`, `"abc"`, `""`).
- `formatMoney(cents)`: format integer cents as `"$1,234.56"`, negatives as `"-$1,234.56"`.
  Throw a `RangeError` for non-integers.
- `allocate(cents, ratios)`: split an integer amount of cents into parts proportional to `ratios`
  (non-negative numbers, at least one positive). The parts are integers that sum exactly to `cents`.
  Each part starts as the rounded-down (floor) share; the leftover cents go one each to the parts with the
  largest fractional remainders, ties going to the earlier index. Negative amounts: allocate the absolute
  value and negate every part. Throw a `RangeError` for invalid ratios.
