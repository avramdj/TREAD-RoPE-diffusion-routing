import os

from beartype import beartype as typechecker
from jaxtyping import jaxtyped

USE_JAX_TYPING = not (os.getenv("DISABLE_JAX_TYPING", "0") == "1")

if USE_JAX_TYPING:
    typed = jaxtyped(typechecker=typechecker)
else:

    def typed(x):
        return x
