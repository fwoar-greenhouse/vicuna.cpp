Write functions for closed integer intervals `[start, end]` with `start <= end`.

- `mergeIntervals(list)`: return a new array of merged intervals sorted by start. Intervals that overlap or
  touch (`[1, 2]` and `[2, 5]`) are merged; `[1, 2]` and `[3, 4]` are not. Do not modify the input.
  Throw a `TypeError` if any interval has `start > end`.
- `insertInterval(list, interval)`: `list` is sorted and already merged. Return a new sorted, merged array
  with `interval` added. Do not modify the input.
