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

Run in Node.js on the CPU of an Intel Mac (2019 MacBook Pro), so the times show how the models
compare with each other, not how fast they are on a GPU. Sizes are the download.

| Model | Weights | Download | Passed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | Time per question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | q4 | 1.0 GB | 3 of 8 | no | yes | no | yes | yes | no | no | no | 1 to 37 s |
| Qwen2.5-0.5B-Instruct | q8 | 1.0 GB | 5 of 8 | yes | yes | yes | yes | yes | no | no | no | 8 to 55 s |
| Qwen2.5-1.5B-Instruct | q4 | 3.1 GB | 6 of 8 | yes | yes | yes | yes | yes | no | yes | no | 20 to 114 s |

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

Qwen2.5-0.5B has not run this test on a GPU; there it writes one checked tool call in 0.1 to 0.7 s.

## Picking a model

- **One-step questions** (a sum, the time, a lookup, a page): Qwen2.5-0.5B at q8 is enough, and it
  is a 1 GB download. Load it with `quant: "q8"`; at q4 it stops calling the skills.
- **Other languages:** Qwen2.5-1.5B picked the right Wikipedia edition and answered in Turkish; the
  0.5B did not.
- **Knowing when not to use a skill:** Hermes 3 (3B), given only its own tool prompt. It needs a
  GPU: on the Intel Mac's CPU it did not finish the test in half an hour.
- **Arithmetic:** the Qwen2.5 models call `calculate`; Hermes 3 does not, and gets sums wrong.
- **Two-step questions:** none of the models above manages one. Larger and tool-trained models are
  next on the list (below).

## To try

- Qwen3 1.7B and 4B, with and without thinking.
- Granite 4.0 1B and 3B.
- The same test on a GPU, for real speeds.
