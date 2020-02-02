import hashlib
from timeit import timeit

def hash_pbkdf2_hmac():
  return hashlib.pbkdf2_hmac('sha256', b'password', b'salt', 100000).hex()

timeit("hash_pbkdf2_hmac()", setup="from __main__ import hash_pbkdf2_hmac", number=1000)
# 108.355244036
# A fast implementation of pbkdf2_hmac is available with OpenSSL. The Python implementation uses an inline version of hmac. It is about three times slower and doesn’t release the GIL.


def hash_sha256():
  return hashlib.sha224(b"sha256").hexdigest()

timeit("hash_sha256()", setup="from __main__ import hash_sha256", number=1000)
# 0.0014648869999973613
# But less secure since there is no pwd and salt.
