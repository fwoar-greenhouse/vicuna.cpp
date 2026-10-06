Write `diffLines(a, b)` where `a` and `b` are arrays of strings (lines).

Return an array of operations `{ op, line }` that turns `a` into `b`:
- `{ op: "=", line }` keeps a line present in both,
- `{ op: "-", line }` deletes a line of `a`,
- `{ op: "+", line }` inserts a line of `b`.

Requirements:
- Reading the `=` and `-` lines in order gives `a`; reading the `=` and `+` lines in order gives `b`.
- The diff is minimal: the number of `=` operations equals the length of a longest common subsequence of
  `a` and `b`.
- Within each run of consecutive changes (between two `=` operations or the ends), all `-` operations come
  before all `+` operations.
- Inputs up to 2000 lines each must finish in well under a second.
