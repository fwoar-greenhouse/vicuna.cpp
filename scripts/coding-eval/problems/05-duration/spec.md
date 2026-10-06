Write functions for durations written like `"1d2h30m15s"`.

- `parseDuration(s)`: return the number of seconds. The units are `d` (86400 s), `h`, `m` and `s`. Each part
  is a non-negative integer followed by a unit. Units must appear at most once and in the order d, h, m, s;
  any subset is allowed (`"90m"`, `"1h5s"`). Values may exceed the next unit (`"90m"` is 5400). Whitespace is
  not allowed. Throw an `Error` for empty strings, unknown units, repeated or out-of-order units, missing
  numbers and missing units.
- `formatDuration(seconds)`: inverse for non-negative integers. Use the largest units first and omit zero
  parts, for example `5400` -> `"1h30m"`, `86401` -> `"1d1s"`. `0` -> `"0s"`. Throw a `RangeError` for
  negative or non-integer input.
