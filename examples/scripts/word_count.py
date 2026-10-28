"""Count the words in a text file and return the most common ones.

Run by the ``taskmaestro.python`` task: arguments come from ``args`` and the
result is written as JSON to ``$TASKMAESTRO_OUTPUT``. Taskmaestro is not imported.
"""

import json
import os
import sys
from collections import Counter

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 3

with open(path, encoding="utf-8") as f:
    text = f.read()
words = [word.strip(".,;:!?\"'()").lower() for word in text.split()]
words = [word for word in words if word]
if not words:
    print(f"No words in {path}", file=sys.stderr)
    sys.exit(1)  # Fails the task (check: true).

with open(os.environ["TASKMAESTRO_OUTPUT"], "w", encoding="utf-8") as f:
    json.dump(
        {
            "words": len(words),
            "lines": len(text.splitlines()),
            "top_words": [word for word, _count in Counter(words).most_common(top)],
        },
        f,
    )
print(f"Counted {len(words)} words in {path}")
