Write `justify(text, width)` that formats text into fully justified lines.

- Words are maximal runs of non-whitespace characters in `text`. Return an array of lines.
- Fill lines greedily: put as many words on a line as fit with single spaces between them.
- Every line except the last is padded to exactly `width` characters by adding spaces between words.
  Distribute extra spaces as evenly as possible; when they do not divide evenly, gaps on the left get one
  more space than gaps on the right. A line with a single word is padded on the right.
- The last line is left-justified: words separated by single spaces, padded on the right to `width`.
- A word longer than `width` goes on its own line, unpadded (that line is longer than `width`).
- If there are no words, return `[]`. Throw a `RangeError` if `width` is not a positive integer.
