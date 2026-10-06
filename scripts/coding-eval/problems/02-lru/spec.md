Write an LRU (least recently used) cache.

`createLRU(capacity)` returns a cache object. `capacity` must be a positive integer; otherwise throw a
`RangeError`. The cache has these methods:

- `get(key)`: return the value for `key`, or `undefined` if it is missing. A hit makes the key the most
  recently used.
- `set(key, value)`: insert or update the key and make it the most recently used. If the cache is now over
  capacity, remove the least recently used key.
- `has(key)`: return true if the key is present. Does not change the order.
- `delete(key)`: remove the key; return true if it was present.
- `size()`: number of keys.
- `keys()`: array of keys from most recently used to least recently used.

Keys can be any value (compare with SameValueZero, like `Map`). All operations should be O(1) except `keys()`.
