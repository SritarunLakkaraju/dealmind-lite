# DealMind - Deal Intelligence Agent (Hindsight)
Run locally:
1. `pip install -r requirements.txt` and copy `.env.example` values into your shell (`export ...`).
2. Hindsight: Cloud (https://ui.hindsight.vectorize.io, promo MEMHACK99) OR `docker run -p 8888:8888 -p 9999:9999 ghcr.io/vectorize-io/hindsight` (needs an LLM key per Hindsight docs).
3. `uvicorn app:app --reload` -> open http://localhost:8000
Demo: play calls 1-4, then Generate brief (compare panels), Coach, Follow-up, then Tactic worked.
Deploy: push to GitHub -> Render/Railway "Web Service" from the Dockerfile, set the 3 env vars.
