# Let's build the GPT Tokenizer


## Links
- [Youtube Video Link](https://www.youtube.com/watch?v=zduSFxRajkE)

## Introduction
- Tokenization might not be one of the most exciting part of working with LLMs, but it’s necessary to understand because many odd behaviors in LLMs trace back to it. 
- In “Let’s Build GPT from Scratch”, Karpathy built a very simple tokenizer (character-level tokenizer) where he mapped each character to an integer token ID, then looked up embeddings from a 65-row table to feed the Transformer. Basically, each token ID indexes a row vector—trainable parameters learned by backprop—which feeds into the Transformer.
- In practice, state-of-the-art models don’t use this character-level tokenization. They use chunk-level tokenization built with algorithms like Byte Pair Encoding (BPE). For example, GPT‑2’s uses a vocabulary of ~50,257 tokens and a context window of 1,024 tokens.
- Tokens are the atomic unit in LLMs: everything is measured in tokens—data volume, context length, training objectives.
- Tokenization is the process that maps text to token sequences and back.
- As mentioned before, many of the issues with LLMs trace back to tokenization. For example all the below issues are related to tokenization:
  - Spelling/character-based tasks can be hard because words may be single multi-character tokens.
  - Simple string operations can be brittle due to arbitrary token boundaries.
  - Non‑English text often expands into more tokens, reducing effective context and hurting performance.
  - Arithmetic can suffer because numbers tokenize inconsistently (sometimes single-token, sometimes split).
  - Early GPT‑2 struggled more with Python in part due to whitespace/tokenization behavior.
  - Trailing whitespace can cause warnings or off‑distribution prompts.
  - Specific strings (e.g., “solidgoldMagikarp”) can map to rare/untrained tokens and trigger bizarre outputs.
  - Token economy matters; formats like YAML may be more token‑efficient than JSON for the same structure.
- We should never brush off tokenization—it’s central to model behavior.

## Tiktokenizer App
- App Link: https://tiktokenizer.vercel.app/

- You can use the tiktokenizer app to visualize how different tokenizers (e.g., GPT‑2 vs `cl100k_base`) split text and count tokens in real time.
- Note, spaces are often part of token chunks; counts include them. Another point to note is how the numbers are tokenized arbitrarily (e.g., 127 may be one token, 677 and 804 may split). This complicates arithmetic for LLMs.
- The same word can tokenize differently at sentence start vs after a leading space; lowercase/uppercase map to different token IDs. The model must learn their equivalence from data.
<img src="images/tiktokenizer_app_1.png" alt="tiktokenizer app" width="400" style="display: block; margin: 0 auto;">

- Non-english text typically use more tokens than equivalent English, effectively “stretching” content within a fixed context window, partly due to tokenizer training skewed toward English. This reduces the effective context and hurts performance on non-english text.
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="images/tiktokenizer_app_lan_gpt2_1.png" alt="tiktokenizer app" width="400">
  <img src="images/tiktokenizer_app_lan_gpt2_2.png" alt="tiktokenizer app" width="400">
</div>

<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="images/tiktokenizer_app_lan_clbase_1.png" alt="tiktokenizer app" width="400">
  <img src="images/tiktokenizer_app_lan_clbase_2.png" alt="tiktokenizer app" width="400">
</div>

- See the number of tokens for the same python code in GPT‑2 and GPT‑4. GPT-2 tokenizer is more wasteful than GPT-4 tokenizer. This is one of the reasons why GPT-2 is not very good with python. OpenAI made a deliberate choice to improve the python or space tokenization for GPT-4. For example, GPT-4 tokenizer (`cl100k_base`) groups a lot more space into a single token, making Python code much denser and improving coding performance due in part to the tokenizer design, not just the model.
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="images/tiktokenizer_app_python_1.png" alt="tiktokenizer app" width="400">
  <img src="images/tiktokenizer_app_python_2.png" alt="tiktokenizer app" width="400">
</div>

- Overall, GPT-4 tokenizer is more efficient than GPT-2 tokenizer even for normal english text. However, just increasing the number of tokens is not strictly better infinitely because as you increase the number of tokens now your embedding table is getting a lot larger. Moreover, at prediction time, you are trying to predict the next token and there's the soft Max there and that grows as well. Thus, we need to have a just right number of tokens in your vocabulary where everything is appropriately dense and still fairly efficient.
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="images/tiktokenizer_app_count_1.png" alt="tiktokenizer app" width="400">
  <img src="images/tiktokenizer_app_count_2.png" alt="tiktokenizer app" width="400">
</div>


## Unicode
- We need a way to convert any string (English, Hindi, emojis) into a sequence of integers to feed into the Transformer's embedding table.
- For that, first we need to understand how are strings (text) represented. In Python, strings are sequences of Unicode code points.
- Here, [Unicode](https://en.wikipedia.org/wiki/Unicode) is a standard defining ~150,000 characters across all the world's scripts. We can get the integer code point for any character using Python's `ord()` function.
- **Why not just use unicode code points as tokens?**:
  - The vocabulary would be very large (~150,000), which makes the embedding and final output layers huge.
  - More critically, the Unicode standard is constantly evolving. A model trained on one version would be incompatible with text using characters from a newer version. It's not a stable, fixed vocabulary.

## UTF-8 and other unicode byte encodings
- Encodings like UTF-8, UTF-16, and UTF-32 are standards for translating Unicode code points into binary byte streams.
- UTF-8 is an encoding that maps code points to a 1–4-byte stream (sequence of bytes). It's the dominant standard on the internet. It’s a variable-length encoding (1-4 bytes per character) and, crucially, is backward-compatible with ASCII. We can convert a Python string to its UTF-8 byte representation using `.encode('utf8')`.
- **Why not use raw UTF-8 bytes as tokens?**:
  - The vocabulary would be tiny (just 256 possible bytes).
  - This would make our token sequences extremely long, as every single byte becomes a token.
  - Long sequences are inefficient and quickly exhaust the Transformer's fixed context length, meaning the model can't see very far back in the text.
- We want to start with the raw UTF-8 byte stream but compress it. We need a method that lets us create a larger, tunable vocabulary to make sequences shorter and denser.
- This is exactly the problem that the Byte Pair Encoding (BPE) algorithm solves. It allows us to compress these byte sequences effectively.
- Links to read more on UTF: 
    - [A programmer's intro to Unicode](https://www.reedbeta.com/blog/programmers-intro-to-unicode/)
    - [UTF-8 Everywhere](https://utf8everywhere.org/)
- Note, we would love to get rid of tokenization entirely and just feed raw byte streams directly into LLMs. There are papers like [MEGABYTE](https://arxiv.org/abs/2305.07185) that are exploring this. This "tokenization-free" approach requires modifying the standard Transformer architecture .While promising, these type of methods are not yet proven at a large enough scale or widely adopted in production models.

## Byte Pair Encoding (BPE)
- Byte Pair Encoding (BPE) is an iterative data compression algorithm. 
- Here's a simple example of BPE:
  We start with a base vocabulary (e.g., the 256 bytes from our UTF-8 stream).
  Suppose the data to be encoded is `aaabdaaabac`.
  - **Initial**: 11 tokens, vocab size 4 (a, b, c, d)
  1. The byte pair "aa" occurs most often, so we replace it with "Z".
     - **Result**: `ZabdZabac` (9 tokens, vocab size 5)
     - **Table**: `Z = aa`
  2. Next, the pair "ab" is most frequent, so we replace it with "Y".
     - **Result**: `ZYdZYac` (7 tokens, vocab size 6)
     - **Table**: `Y = ab`, `Z = aa`
  3. We could continue, replacing "ZY" with "X".
     - **Result**: `XdXac` (5 tokens, vocab size 7)
     - **Table**: `X = ZY`, `Y = ab`, `Z = aa`
  After each step, our sequence of tokens gets shorter, but our vocabulary size grows by one.
- To tokenize new text, we first convert it to a UTF-8 byte stream and then greedily apply the learned merge rules in order. To decode, we reverse the process, replacing merged tokens with their constituent byte pairs until we're back to the original byte stream.

  
### Training the Tokenizer

#### Step 1: Initial Setup: 
- We start with a representative sample of text (the longer, the better for statistics) and encode it into a stream of UTF-8 bytes. For easier manipulation, we convert this byte stream into a list of integers (0-255). This is our initial token sequence.

#### Step 2: Define Core Functions:
- We define two core functions:
  - **`get_stats()`**: A function that iterates through the current token list and returns a dictionary of counts for all adjacent pairs.
  - **`merge()`**: A function that takes a token list, a specific pair (like `(101, 32)`), and a new token ID (like `256`). It then returns a new token list where every occurrence of the pair is replaced by the new ID.

#### Step 3: The Iterative Training Loop:
- We choose a target vocabulary size as a hyperparameter (e.g., 276). The number of merges to perform.
> Note: The number of times we perform this merge operation is a key hyperparameter. More merges lead to a larger vocabulary and shorter, more compressed sequences. Finding the right balance (e.g., GPT-4's ~100,000 token vocabulary seems to a good balance for LLMs) is a trade-off between sequence length and model complexity.
- We loop for the desired number of merges. In each iteration:
    - Use `get_stats()` on the current token list to find the most frequent pair.
    - "Mint" a new token ID (the next available integer, starting at 256).
    - Use `merge()` to replace all occurrences of the most frequent pair with the new token ID.
    - Store the merge rule (e.g., `(101, 32) -> 256`) in a `merges` dictionary. This dictionary *is* our trained tokenizer vocabulary.

>Note: It's important to note that newly created tokens (like 256) are immediately eligible for being part of a new pair in the next iteration. This is how BPE builds up tokens representing longer and longer character sequences, creating a "forest" of merge trees.

- After the loop completes, we have our trained `merges` vocabulary. `merges` dictionary is our trained tokenizer vocabulary.

- We can also measure the effectiveness of our tokenizer by calculating the **compression ratio**: the length of the original byte sequence divided by the length of the final token sequence. More merges will lead to a higher compression ratio and a larger vocabulary. Finding the right balance is a key hyperparameter trade-off.

### Tokenization is a separate stage
- The tokenizer is a completely separate entity from the LLM. Training the tokenizer is a distinct, one-time pre-processing stage that happens *before* LLM training.
- The tokenizer is trained on its own corpus of text, which can be different from the dataset used to train the LLM. The composition of this training data is critical.
- The mix of languages and content (e.g., code, prose) in the tokenizer's training set determines its efficiency. For example, including a large amount of Japanese text will lead to more Japanese-specific merges, resulting in better compression (shorter token sequences) for Japanese. This is beneficial for the LLM, which has a finite context window.
- Once trained, the tokenizer acts as a translator. It has two main jobs:
    - **Encoding**: Converting raw text into a sequence of token IDs.
    - **Decoding**: Converting a sequence of token IDs back into raw text.
<img src="images/tokenization.png" alt="tokenization" width="500" style="display: block; margin: 0 auto;">

- A common workflow is to use the trained tokenizer to pre-process the entire LLM training dataset. All the text is converted into a massive sequence of tokens, which is then saved to disk. The LLM then trains directly on these token sequences, often without ever seeing the raw text again.

### Decoder
- We first create a `vocab` dictionary that maps every token ID to its corresponding byte sequence.
    - We initialize it with the first 256 entries, mapping integers `0` through `255` to their single-byte representations.
    - We then iterate through our trained `merges` dictionary *in the exact order they were created*. For each merge `(p1, p2) -> new_id`, we define the byte sequence for `new_id` as the concatenation of the byte sequences for `p1` and `p2`. Note, since python3.7, the order of dictionary items is guaranteed to be the same as the order of insertion.
- We then decode the token IDs into a human-readable Python string using the `vocab` dictionary.
- Finally, we convert the bytes to standard Python string using the `.decode('utf8')` method.
- A crucial detail is that not all possible byte sequences are valid UTF-8. An LLM could potentially output a sequence of tokens that is invalid.
    - By default, `.decode()` will throw a `UnicodeDecodeError`.
    - To prevent this, we use the `errors='replace'` argument: `.decode('utf8', errors='replace')`. This will insert a special replacement character () for any invalid byte sequences, making the decoding process robust and signaling that the LLM produced a malformed output.

### Encoder
- We start by taking the input text, encoding it into UTF-8, and converting the resulting byte stream into a list of integers (0-255). This is our initial token sequence.
- We repeatedly merge pairs in the token sequence based on our trained `merges` vocabulary.
    - In each step of the loop, we find the pair of adjacent tokens in our current sequence that has the *lowest rank* (i.e., was learned earliest) in our `merges` dictionary.
    - We then replace this single, highest-priority pair with its corresponding new token ID.
    - We continue this process, one merge at a time, until there are no more mergeable pairs in the sequence according to our vocabulary.
- Note, the implementation needs to handle short sequences (fewer than two tokens) where no merges are possible.
- A key property to check is round-trip consistency. For any given text, `decode(encode(text))` should return the original text. This confirms our implementation is working correctly. The reverse is not necessarily true, as not all token sequences are valid.

## Forced splits using the regex patterns (GPT series)
- A naive BPE implementation might merge words with punctuation (e.g., creating a single token for `"dog."`, another for `"dog?"`, etc). It is not the best use of the vocabulary. This is suboptimal because it mixes semantics with formatting.
- To prevent this, the GPT-2 tokenizer introduces a pre-processing step. Before BPE is applied, the text is first split into chunks using a complex regular expression pattern. 
- The final token sequence is the concatenation of the results from each chunk. This effectively creates "forced splits" and prevents merges across different categories (e.g., a letter will never merge with a punctuation mark from an adjacent chunk). The whitespace handling is also very subtle, designed to keep leading spaces attached to words.
- Note: OpenAI never released the training code for the GPT-2 tokenizer, only the inference code. We can observe that there are likely additional undocumented rules (e.g., spaces are never merged, even when the regex would allow it). Thus, the training process i not just Chunking followed by BPE.

### GPT-2 Tokenizer Regex Pattern
```python
r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```
- It pre-split raw text into chunks **before** BPE so merges never cross category boundaries (letters vs numbers vs punctuation vs whitespace). Final tokens = BPE applied **inside each chunk** then concatenated. 
- Contractions:
    ```
    's | 't | 're | 've | 'm | 'll | 'd
    ```
    - It matches frequent English clitics right after a word: `it's`, `we're`, `I'm`, `you'll`, etc.
    - This ensures that these tiny morphemes are their **own chunks** so BPE learns/reuses them across contexts instead of absorbing them inconsistently into words/punctuation.
    - Note, only ASCII `'` + lowercase are matched
- Letters with optional leading space
    ```
     ?\p{L}+
    ```
    - It matches one optional leading space + one or more **letters** (Unicode).
    - This keeps words in a letter-only bucket, and deliberately glue a single leading space to the word so `" hello"` is a unit. That encourages stable `"␠word"` merges and keeps inter-word spacing regular. 
- Numbers with optional leading space
    ```
     ?\p{N}+
    ```
    - It matches one optional leading space + one or more **digits/numeric** characters (Unicode).
    - This prevents alphanumeric blends from merging across the letter/number boundary (`"foo123"` → `"foo" | "123"`). This avoids polluting the vocab with accidental word+number tokens.
- Other (non-space, non-letter, non-number) with optional leading space
    ```
     ?[^\s\p{L}\p{N}]+
    ```
    - It matches one optional leading space + one or more **punctuation/symbols/emoji** bytes.
- Whitespace (all but the last char in a run)
    ```
    \s+(?!\S)
    ```
    - It matches any remaining whitespace (e.g., string-end).
    - This ensures nothing falls through; leftover trailing spaces become their own chunks. (Empirically, spaces weren’t merged in the learned vocab; likely an additional rule beyond this regex). 

- Whitespace (catch-all)
   ```
   \s+
   ```
   - It matches any remaining whitespace (e.g., string-end).

## tiktoken
- [tiktoken](https://github.com/openai/tiktoken) is OpenAI's open-source library for fast BPE tokenization. It's important to note that this is for *inference* only; you cannot use it to train a new tokenizer.
- It exposes different tokenizers, including GPT-2's and GPT-4's (`cl100k_base`). We can see key differences, such as GPT-4's more efficient merging of whitespace characters.
- The pre-tokenization regex pattern used to force splits was updated for GPT-4. We can inspect these patterns in the `tiktoken` library source files. 
    - GPT-2 regex pattern: 
        ```python
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
        ```
    - GPT-4 regex pattern:
        ```python
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
        ```
- Key changes include:
    - Fixes the issue where contractions were treated differently based on capitalization.
    - The new pattern only allows numbers of up to three digits to be merged, preventing the creation of excessively long number tokens.
    - The full rationale for these changes is not publicly documented by OpenAI.
- The core GPT-2's [encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py) BPE logic in their file is algorithmically identical to the greedy merging loop implemented in the notebook.

## Special Tokens
- In addition to the tokens generated by BPE, we can add "special tokens" to our vocabulary. These are used to signal structure or metadata to the LLM, like the end of a document.
- GPT-2's vocabulary has 50,257 tokens. This comes from 256 base byte tokens, 50,000 learned merges, and **one special token**: `<|endoftext|>`. This token is used to separate documents in the training data, signaling to the model where one piece of content ends and another begins. The model must learn this from data.
- Special tokens are handled *outside* the normal BPE process. The tokenizer has special-case logic (e.g., in `tiktoken`) that looks for these exact strings and replaces them with their assigned ID. They are not generated from byte merges.
- Special tokens are essential for fine-tuning, especially for creating chat models. Tokens like `<|im_start|>` and `<|im_end|>` are used to delimit messages between a user and an assistant, providing a clear structure for conversations.
- The GPT-4 tokenizer expanded the set of special tokens to include ones for tasks like "fill in the middle" (FIM), which are useful for code completion and other specific tasks. You can see them in the [tiktoken](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py) library source files.
- We can extend an existing tokenizer with our own custom special tokens. For example, see the [tiktoken example](https://github.com/openai/tiktoken/tree/main?tab=readme-ov-file#extending-tiktoken). 
- However, if we do this, it also requires "model surgery" especially if we are doing fine-tuning:
    1.  The token embedding matrix in the Transformer must be extended by adding a new row for each new token.
    2.  The final output projection layer (LM head) must also be extended to predict the new tokens.
    3.  These new weights are typically initialized randomly and then fine-tuned so the model learns their meaning.