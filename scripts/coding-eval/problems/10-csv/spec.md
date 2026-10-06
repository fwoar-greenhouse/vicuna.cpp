Write `parseCSV(text)` that parses CSV (RFC 4180 style) into an array of rows, each an array of strings.

- Fields are separated by `,` and rows by `\n` or `\r\n`.
- A field may be quoted with `"`. Inside quotes, commas and line breaks are part of the field and `""`
  stands for one `"`. Quotes are only special at the start of a field.
- Unquoted fields are taken as-is (no trimming); a `"` inside an unquoted field is a normal character.
- A single trailing line break at the end of the text does not create an extra row. Empty lines in the
  middle are rows with one empty field `[""]`.
- `parseCSV("")` returns `[]`.
- Throw a `SyntaxError` for an unterminated quoted field, or for characters other than `,` or a line break
  right after a closing quote.
