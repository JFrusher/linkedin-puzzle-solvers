# 0004: Run locally; no hosted service

**Context.** The project could have been a hosted web service. It is used by one person on their own
screenshots, with large models (EasyOCR, Sentence-BERT) and CPU-heavy solves.

**Decision.** Everything runs on the user's machine: the CLI, the Streamlit playground, and a local server for
the browser extension. The server binds to `127.0.0.1` by default (`--host` overrides it) and handles one request at a time, with a 30-second socket
deadline.

**Why.** No data leaves the machine, there is no hosting cost, and the Docker image bakes in every model so it
starts offline. One request at a time is also required: solves profile memory with `tracemalloc`, which cannot
run twice at once.

**Cost.** No public demo; the README's GIFs and screenshots do that job. The server has no login, so binding it
to anything other than localhost would expose it.
