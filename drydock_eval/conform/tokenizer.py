# Copyright 2026 The drydock-eval Authors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenizers for the template-identity check.

* :class:`HFTokenizer` loads a user-supplied Hugging Face ``tokenizer.json`` (needs the ``tokenizers`` package).
  The kit never ships a model tokenizer.
* :class:`ToyTokenizer` is a small deterministic tokenizer used by the tests, the mock server and the mutation suite.
  Like the Gemma tokenizer it encodes ``"\\n\\n"`` as one token, so a thought that ends with a newline renders to the
  same number of tokens as one that does not (only one token id changes). That property is why a count-only
  template check cannot see the history-thought newline drift.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

# Turn and channel markers of the public Gemma 4 prompt format, used by the toy template.
TOY_SPECIAL_TOKENS: tuple[str, ...] = (
    "<bos>", "<eos>", "<pad>", "<|turn>", "<turn|>", "<|think|>", "<|channel>", "<channel|>",
    "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", '<|"|>',
)


class Tokenizer(Protocol):
    name: str
    sha256: str

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class ToyTokenizer:
    """Deterministic toy tokenizer: special markers, ``\\n\\n``, ``\\n``, words with an optional leading space,
    digits, space runs and single other characters. Ids are stable across processes, and every id it produces
    decodes in any instance: a single character's id is its code point plus an offset, and a longer token's id holds
    the token's own UTF-8 bytes (so word ids are large integers). Version 2 hashed words, so a fresh instance could
    not decode a word it had not encoded itself."""

    name = "drydock-eval-toy-tokenizer-v3"

    _NEWLINE2_ID = 900
    _CHAR_BASE = 1_000
    _WORD_BASE = 2_000_000

    def __init__(self) -> None:
        specials = sorted(TOY_SPECIAL_TOKENS, key=len, reverse=True)
        self._special_ids = {tok: i + 1 for i, tok in enumerate(TOY_SPECIAL_TOKENS)}
        pattern = "|".join(re.escape(s) for s in specials)
        self._re = re.compile(rf"{pattern}|\n\n|\n| ?[A-Za-z]+|\d|[ \t]+|.", re.DOTALL)
        self._cache: dict[str, int] = dict(self._special_ids)
        self._cache["\n\n"] = self._NEWLINE2_ID
        self._reverse: dict[int, str] = {v: k for k, v in self._cache.items()}
        self.sha256 = hashlib.sha256((self.name + "|" + pattern).encode()).hexdigest()

    def _new_id(self, tok: str) -> int:
        if len(tok) == 1:
            tid = self._CHAR_BASE + ord(tok)
        else:
            # a 0x01 lead byte keeps the mapping one-to-one and every such id above _WORD_BASE + 2**16
            tid = self._WORD_BASE + int.from_bytes(b"\x01" + tok.encode("utf-8"), "big")
        self._cache[tok] = tid
        self._reverse.setdefault(tid, tok)
        return tid

    def _token_of(self, tid: int) -> str | None:
        """The token an id stands for, whether or not this instance has encoded it (None: not a toy id)."""
        if self._CHAR_BASE <= tid < self._WORD_BASE:
            try:
                return chr(tid - self._CHAR_BASE)
            except (ValueError, OverflowError):
                return None
        if tid > self._WORD_BASE:
            n = tid - self._WORD_BASE
            raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
            if len(raw) >= 3 and raw[0] == 1:
                try:
                    tok = raw[1:].decode("utf-8")
                except UnicodeDecodeError:
                    return None
                if len(tok) > 1 and tok not in self._special_ids and self._re.fullmatch(tok):
                    return tok
        return None

    def tokens(self, text: str) -> list[str]:
        return self._re.findall(text)

    def encode(self, text: str) -> list[int]:
        cache = self._cache
        new = self._new_id
        return [cache[t] if t in cache else new(t) for t in self._re.findall(text)]

    def decode(self, ids: Sequence[int]) -> str:
        rev = self._reverse
        out = []
        for i in ids:
            tok = rev.get(i)
            if tok is None:
                tok = self._token_of(i) if isinstance(i, int) else None
                if tok is None:
                    tok = f"<unk:{i}>"
            out.append(tok)
        return "".join(out)


class HFTokenizer:
    """A Hugging Face ``tokenizer.json`` loaded with the ``tokenizers`` package. Chat prompts are encoded without
    adding special tokens, because the chat template already writes the BOS marker (as vLLM does for chat requests)."""

    def __init__(self, path: str) -> None:
        try:
            from tokenizers import Tokenizer as _Tok  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError("loading tokenizer.json needs the 'tokenizers' package (pip install 'drydock-eval[hf]')") from exc
        with open(path, "rb") as fh:
            raw = fh.read()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.name = f"tokenizer.json sha256:{self.sha256[:16]}"
        self._tok = _Tok.from_str(raw.decode("utf-8"))

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)


def load_tokenizer(spec: str) -> Tokenizer:
    """``spec`` is ``"toy"`` or a path to a Hugging Face ``tokenizer.json``."""
    if spec == "toy":
        return ToyTokenizer()
    return HFTokenizer(spec)
