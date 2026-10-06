Write semantic version helpers (versions look like `1.2.3`, `1.2.3-alpha.1`, `1.2.3+build.5`).

- `compare(a, b)`: return -1, 0 or 1 using SemVer 2.0.0 precedence. Build metadata (after `+`) is ignored.
  A pre-release version is lower than the same version without one. Pre-release identifiers are compared
  left to right: numeric identifiers numerically, others in ASCII order, numeric lower than non-numeric,
  and a shorter list is lower if all earlier identifiers are equal. Throw a `TypeError` for invalid versions
  (missing parts, leading zeros in numbers, empty identifiers).
- `satisfies(version, range)`: a range is one or more comparator sets joined by `||`; a comparator set is
  one or more comparators separated by spaces, all of which must match. Comparators:
  - `1.2.3` or `=1.2.3`: equal
  - `>1.2.3`, `>=1.2.3`, `<1.2.3`, `<=1.2.3`
  - `^1.2.3`: `>=1.2.3 <2.0.0`; `^0.2.3`: `>=0.2.3 <0.3.0`; `^0.0.3`: `>=0.0.3 <0.0.4`
  - `~1.2.3`: `>=1.2.3 <1.3.0`
  Comparator versions are always full `major.minor.patch` (optionally with a pre-release).
  A version with a pre-release only satisfies a comparator set if one of its comparators has a pre-release
  on the same `major.minor.patch` (so `1.3.0-beta` does not satisfy `^1.2.0`, but `1.2.3-beta.2`
  satisfies `>=1.2.3-beta.1`).
