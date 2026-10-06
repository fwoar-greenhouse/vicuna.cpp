Write `evaluate(expr)`, an arithmetic expression evaluator. Do not use `eval` or `Function` (they are not
available).

- Numbers: decimal literals such as `3`, `0.5`, `.5`, `10.25` (no exponent notation).
- Operators: binary `+ - * / ^`, unary `-` and `+`, parentheses. Whitespace is allowed anywhere between tokens.
- Precedence, high to low: `^` (right associative), unary `-`/`+`, `*` and `/` (left associative),
  `+` and `-` (left associative). So `-2^2` is `-4`, `2^3^2` is `512`, and `2*-3` is `-6`.
- Throw a `SyntaxError` for malformed input (empty input, unbalanced parentheses, unknown characters,
  missing operands, two numbers in a row).
- Throw a `RangeError` for division by zero.
