# Which model for skills

How the chat models we tried handle Kiln's built-in skills (`skills()` with `runTools`): which
questions each gets right, and how long it takes. Use it to pick a model; add a row when you try
another.

## The test

Eight questions, each asked once in a fresh chat with all four skills offered, the
`SKILL_INSTRUCTIONS` system message (unless a row says otherwise), temperature 0 and the live network. A question passes when
the model calls the right skill with usable arguments and its answer agrees with what the skill
returned.

| # | Question | What a pass needs |
| --- | --- | --- |
| 1 | What is 1234 * 5678 + 99? | `calculate`, answer 7006751 |
| 2 | What time is it in Tokyo right now? | `get_date_time` for Tokyo |
| 3 | What day of the week is it today? | `get_date_time` |
| 4 | Who was Ada Lovelace? Look it up on Wikipedia. | `search_wikipedia` |
| 5 | Read https://example.com and tell me what the page says. | `read_web_page` |
| 6 | How tall is Mount Everest, and what is that height divided by 2? | two steps: a lookup, then `calculate` (4424 m) |
| 7 | Kapadokya nedir? Türkçe Vikipedi'ye bak. | `search_wikipedia` in Turkish, answer in Turkish |
| 8 | Say hello in French and in Turkish. | no skill: just the answer |

Questions 1 to 5 and 7 are one step. Question 6 needs two skills in a row, and question 8 needs the
model to see that no skill fits: these two separate the models.

## Results

Run in Node.js on the CPU of an Intel Mac (2017 MacBook Pro), so the times show how the models
compare with each other, not how fast they are on a GPU. Sizes are the download.

| Model | Weights | Download | Passed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | Time per question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | q4 | 1.0 GB | 3 of 8 | no | yes | no | yes | yes | no | no | no | 1 to 37 s |
| Qwen2.5-0.5B-Instruct | q8 | 1.0 GB | 5 of 8 | yes | yes | yes | yes | yes | no | no | no | 8 to 55 s |
| Qwen2.5-1.5B-Instruct | q4 | 3.1 GB | 6 of 8 | yes | yes | yes | yes | yes | no | yes | no | 20 to 114 s |

| Granite 4.0 1B, no system message | Q8_0 GGUF | 1.7 GB | 8 of 8 | yes | yes | yes | yes | yes | yes | yes | yes | 5 to 127 s |
| Granite 4.0 1B, with `SKILL_INSTRUCTIONS` | Q8_0 GGUF | 1.7 GB | 7 of 8 | yes | yes | yes | yes | yes | yes | no | yes | 6 to 141 s |
| FunctionGemma 270M, its own developer prompt only | f32 | 0.5 GB | 5 of 8 | yes | yes | yes | yes | yes | no | no | no | 3 to 17 s |
| Qwen3 4B, no system message, thinking off, `maxSeq` 2048 | Q4_K_M GGUF | 2.5 GB | 7 of 8 | yes | yes | no | yes | yes | yes | yes | yes | see below (GPU) |
| Hermes 3 (Llama 3.2 3B), its own tool prompt only | Q4_K_M GGUF | 2.0 GB | 6 of 8 | no | yes | yes | yes | yes | no | yes | yes | see below (GPU) |
| Hermes 3 (Llama 3.2 3B), with `SKILL_INSTRUCTIONS` | Q4_K_M GGUF | 2.0 GB | 3 of 8 | no | yes | yes | – | no | no | yes | no | see below (GPU) |

"–": not judged; Wikipedia refused the request that time (HTTP 429).

What went wrong, by model:

- **Qwen2.5-0.5B, q4.** Answers arithmetic and the weekday from memory, wrongly (709155; "Monday").
  Question 7: it searches the English Wikipedia and answers in English. Question 8: it searches
  Wikipedia for "hello".
  Without the system message it mostly asks the user for the information instead of calling.
- **Qwen2.5-0.5B, q8.** The one-step questions work. Question 6: it writes a call with an endless
  number and runs out of tokens. Question 7: it reads the English article and answers in broken
  Turkish. Question 8: it calls `calculate("print('Bonjour')")`.
- **Qwen2.5-1.5B, q4.** The one-step questions work, Turkish included. Question 6: one call,
  `2 * 8848`, so it doubles a height it recalled instead of looking it up and halving. Question 8:
  it searches Wikipedia for "hello French" and reports the start-up it found.

- **Granite 4.0 1B, no system message.** Nothing went wrong: the only model to pass every
  question. For question 6 it looks Mount Everest up, reads 8,848.86 m and halves it itself,
  correctly (4,424.43 m), without calling `calculate`; question 7 it searches the Turkish Wikipedia
  and answers in Turkish; question 8 it just answers.
- **Granite 4.0 1B, with `SKILL_INSTRUCTIONS`.** The same, except question 7: it searches the
  English Wikipedia, and its Turkish answer has errors. The system message does not help it.
- **FunctionGemma 270M** (Google's Gemma 3 270M tuned for function calling; no system message, so
  its template writes the developer line it was trained with). The smallest model here, and every
  one-step question in English works, arithmetic included. Question 3 it answers with Tokyo's date,
  not the local one. Question 6: one call, `2 * Everest height / 2`. Question 7: no call, and a
  Turkish sentence that says nothing. Question 8: it reads `example.com/french`. It writes calls in
  a format of its own (`<start_function_call>call:name{…}`), which Kiln reads but does not check
  against the skills' schemas. Google's repo asks for a sign-in; the test used Unsloth's copy.
- **Qwen3 4B, no system message, thinking off.** Question 3: no call; it says it cannot know the
  date and offers to check. Question 6 counts as a pass on the answer (4,424 m), but it skipped the
  lookup: it calls `calculate("8848 / 2")` with a height it recalled. The rest is right, Turkish
  included, and question 8 it just answers.
- **Qwen3 4B with thinking on** was run once but not scored: the test printed only the start of
  each thinking block. It called `get_date_time` for question 3, made no call at all for
  question 6, and took four to six times as long.
- **Hermes 3, its own tool prompt only** (Kiln writes the prompt Hermes was trained with; no other
  system message). The only model here that answers question 8 without reaching for a skill, and
  the lookups and the web page work. It never calls `calculate`: question 1 it works out in its
  head, wrongly (6997421), and question 6 it answers from memory in prose, without dividing.
- **Hermes 3, with `SKILL_INSTRUCTIONS` in front of its own prompt.** Worse on every count: it
  writes "calculate result=1234 * 5678 + 99" as text instead of a call, and for question 5 it calls
  nothing and makes up the page's text. Do not give Hermes that system message.

## On a GPU

On an M3 MacBook Air (16 GB, macOS 15, Node.js through Dawn):

| Model | Load | Time per question |
| --- | --- | --- |
| Hermes 3 (Llama 3.2 3B), Q4_K_M | 12 to 25 s from disk | 2 to 16 s, the skill's network request included |
| Granite 4.0 1B, Q8_0, no system message | 10 s from disk | 0.6 to 11.5 s, the skill's network request included; all eight in 58 s |
| Qwen3 4B, Q4_K_M, `maxSeq` 2048, thinking off | 9 s from disk | 2.3 to 20.4 s; all eight in 94 s |
| Qwen3 4B, the same with thinking on | 9 s from disk | 20 to 54 s; all eight in 5 min 40 s |

Granite passed all eight again there, with another test using the machine at the same time. One
slip in its wording: for Tokyo it wrote "00:47 (6:47 AM)".

Qwen2.5-0.5B has not run this test on a GPU; there it writes one checked tool call in 0.1 to 0.7 s.

## Memory

- Kiln's CPU backend holds at most 4 GiB. Qwen3 4B (Q4_K_M) needs slightly more, so it runs on a
  GPU only.
- On the 16 GB MacBook Air, Qwen3 4B with the default context filled the machine's swap and was
  stopped before its first answer. With `maxSeq: 2048` it ran, and swap still grew by about 2 GB
  per run (another test was using the machine). Give it a small `maxSeq` on a 16 GB machine.

## Picking a model

- **First choice: Granite 4.0 1B** (`ibm-granite/granite-4.0-1b-GGUF/granite-4.0-1b-Q8_0.gguf`,
  1.7 GB, Apache 2.0), without a system message. It is the only model that passed all eight
  questions, the two-step one and the one that needs no skill included. It is slow on a CPU; on a
  GPU it answers in 1 to 12 s.
- **Qwen3 4B** is no better here (7 of 8), slower, larger, and needs a GPU and a small `maxSeq`.
- **Smallest: FunctionGemma 270M**, half a gigabyte, for one-step questions in English (a sum, the
  time, a lookup, a page). Qwen2.5-0.5B does as much at q8 (1 GB; at q4 it stops calling skills).
- **System message:** `SKILL_INSTRUCTIONS` helps the small Qwen2.5 models, which otherwise ask the
  user instead of calling. Leave it out for Granite 4.0, Hermes 3 and FunctionGemma: each does
  better with the prompt it was trained on.
- **Arithmetic:** Hermes 3 never calls `calculate` and gets sums wrong; the others call it.
- Not yet known: how any of these does on harder chains (three steps or more), and GPU speeds for
  the Qwen2.5 models and FunctionGemma.

## To try

- Qwen3 1.7B; Qwen3 4B with thinking on, scored properly.
- Granite 4.0 3B (micro).
- FunctionGemma is meant to be fine-tuned on an application's own functions; untuned is what was tried.
- The same test on a GPU, for real speeds.
