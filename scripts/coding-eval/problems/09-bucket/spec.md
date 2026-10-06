Write a token bucket rate limiter. The clock is passed in; do not use `Date` or timers.

`createBucket({ capacity, refillPerSecond, now })`:
- `capacity`: maximum tokens, a positive number. The bucket starts full.
- `refillPerSecond`: tokens added per second, a non-negative number. Tokens accrue continuously
  (fractions count) and never exceed `capacity`.
- `now`: a function returning the current time in milliseconds. Time never goes backwards.
- Throw a `RangeError` for invalid options (capacity <= 0, negative refill rate, missing `now`).

The returned object has:
- `tryTake(n = 1)`: if at least `n` tokens are available, remove them and return `true`; otherwise return
  `false` and change nothing. Throw a `RangeError` if `n` is not a positive number or is greater than `capacity`.
- `available()`: the current number of tokens (may be fractional).
- `msUntil(n = 1)`: milliseconds until `n` tokens will be available (0 if they already are). Return
  `Infinity` if the refill rate is 0 and there are not enough tokens.
