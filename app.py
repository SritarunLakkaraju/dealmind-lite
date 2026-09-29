"""DealMind - Deal Intelligence Agent on Hindsight memory.
retain() every call -> recall()/reflect() before each agent action -> feedback loop teaches the playbook."""
import os, json, asyncio, pathlib, logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from hindsight_client import Hindsight
from groq import AsyncGroq
from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger("dealmind")
BASE = pathlib.Path(__file__).parent
DATA = json.loads((BASE / "data/deals.json").read_text())
DEAL = DATA["deal"]
MODELS = ["openai/gpt-oss-120b", "qwen/qwen3-32b"]      # fallback order
MISSION = ("I am a deal-intelligence memory for an enterprise AI sales team. Track stakeholders, objections, "
           "requirements, competitors, commitments and which tactics won or lost deals.")

HS_URL = os.getenv("HINDSIGHT_URL") or (
    "https://api.hindsight.vectorize.io" if os.getenv("HINDSIGHT_API_KEY") else "http://localhost:8888")
hs = Hindsight(base_url=HS_URL, api_key=os.getenv("HINDSIGHT_API_KEY") or None, timeout=45.0)
print("Hindsight URL:", HS_URL, "| key set:", bool(os.getenv("HINDSIGHT_API_KEY")))
llm = AsyncGroq(api_key=os.getenv("GROQ_API_KEY", "missing"))

@asynccontextmanager
async def lifespan(app):
    try:
        yield
    finally:
        await hs.aclose()
        await llm.close()

app = FastAPI(title="DealMind", lifespan=lifespan)

STATE = {"run": 1, "played": [], "degraded": False, "seeded": False}
LOCAL: dict[str, list[str]] = {}          # fallback store if Hindsight is unreachable

def bank(): return f"{DEAL['id']}-r{STATE['run']}"    # per-deal bank
PLAYBOOK = "sales-playbook"                            # shared cross-deal bank

async def retry(fn, tries=3, base=0.7):
    for i in range(tries):
        try: return await fn()
        except Exception as e:
            log.warning("attempt %s failed: %s", i + 1, e)
            if i == tries - 1: raise
            await asyncio.sleep(base * 2 ** i)

async def ensure_banks():
    for bid in (bank(), PLAYBOOK):
        try: await hs.acreate_bank(bank_id=bid, name=bid, mission=MISSION,
                                   disposition={"skepticism": 4, "literalism": 3, "empathy": 3})
        except Exception as e: log.info("create_bank %s: %s (likely exists)", bid, e)
    if not STATE["seeded"]:
        try:
            await retry(lambda: hs.aretain_batch(bank_id=PLAYBOOK, document_id="past-deals", items=[
                {"content": f"{p['deal']} ({p['outcome']}): {p['note']}", "context": "closed deal post-mortem"}
                for p in DATA["playbook"]]))
            STATE["seeded"] = True
        except Exception: STATE["degraded"] = True

async def remember(bid, text, **kw):
    LOCAL.setdefault(bid, []).append(text)             # always mirror locally
    try: await retry(lambda: hs.aretain(bank_id=bid, content=text, **kw)); return True
    except Exception: STATE["degraded"] = True; return False

async def recall_text(bid, query, max_tokens=2500):
    try:
        r = await retry(lambda: hs.arecall(bank_id=bid, query=query, budget="mid", max_tokens=max_tokens))
        return [f"[{x.type}] {x.text}" for x in r.results]
    except Exception:
        STATE["degraded"] = True
        return [f"[local] {t[:400]}" for t in LOCAL.get(bid, [])[-6:]]   # graceful degradation

async def ask_llm(system, user):
    last = None
    for m in MODELS:                                   # model fallback + retry on malformed/empty output
        for _ in range(2):
            try:
                out = await llm.chat.completions.create(model=m, temperature=0.3, max_tokens=700,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                txt = (out.choices[0].message.content or "").strip()
                if txt: return txt
            except Exception as e: last = e; await asyncio.sleep(0.5)
    return f"(LLM unavailable: {last}) Raw memory shown instead."

@app.get("/")
def home(): return FileResponse(BASE / "static/index.html")

@app.get("/api/deal")
def deal():
    return {"company": DEAL["company"], "product": DEAL["product"], "state": STATE,
            "calls": [{"n": c["n"], "title": c["title"], "date": c["date"], "transcript": c["transcript"]} for c in DEAL["calls"]]}

@app.post("/api/reset")
def reset():                                           # fresh bank = clean before/after demo
    STATE.update(run=STATE["run"] + 1, played=[], degraded=False); return {"ok": True}

@app.post("/api/play/{n}")
async def play(n: int):
    await ensure_banks()
    c = next((c for c in DEAL["calls"] if c["n"] == n and n < 5), None)
    if not c: return {"ok": False, "error": "Call 5 is the upcoming call - use Brief."}
    ok = await remember(bank(), f"Call {n} - {c['title']} with {DEAL['company']} on {c['date']}\n{c['transcript']}",
                  context="sales call transcript", document_id=f"call-{n}",
                  timestamp=datetime.fromisoformat(c["date"]), metadata={"deal": DEAL["id"], "call": str(n)})
    if n not in STATE["played"]: STATE["played"].append(n)
    return {"ok": True, "stored_in_hindsight": ok, "degraded": STATE["degraded"]}

BRIEF_Q = ("Prepare the rep for the next call: each stakeholder's role and concerns, unresolved objections, hard "
           "requirements, competitor, commitments we made, and budget owner.")

@app.get("/api/brief")
async def brief(memory: int = 1):
    if not memory or not STATE["played"]:
        return {"text": await ask_llm("You are a sales-prep assistant. 5 bullets max.",
                "Brief me for my next call with Kestrel Logistics. I have no notes."), "memory": False}
    try:
        result = await retry(lambda: hs.areflect(bank_id=bank(), query=BRIEF_Q, budget="mid",
            context="rep is about to join call 5 with the CFO; answer as a concise bulleted brief"))
        return {"text": result.text, "memory": True}
    except Exception:
        STATE["degraded"] = True
        mem = "\n".join(await recall_text(bank(), BRIEF_Q))
        return {"text": await ask_llm("You are a sales-prep assistant. Concise bullets, use only the memory.", mem), "memory": True, "fallback": True}

@app.get("/api/coach")
async def coach(objection: str = "CFO wants payback proof and dislikes big upfront commitments"):
    deal_mem = await recall_text(bank(), objection)
    play_mem = await recall_text(PLAYBOOK, objection)  # cross-deal learning
    return {"text": await ask_llm("You are a sales coach. Recommend 2-3 tactics. Cite which past deal or call each comes from. "
            "Say what NOT to do if a past deal was lost that way.",
            f"Objection: {objection}\n\nThis deal:\n" + "\n".join(deal_mem) + "\n\nPast deals:\n" + "\n".join(play_mem)),
            "sources": {"deal": deal_mem, "playbook": play_mem}}

@app.get("/api/followup")
async def followup():
    mem = await recall_text(bank(), "commitments we made, requirements, stakeholder names, next steps")
    return {"text": await ask_llm("Write a short follow-up email from Arjun to Priya. Reference specific numbers, names and "
            "commitments from memory. No fluff.", "\n".join(mem))}

class Feedback(BaseModel):
    tactic: str
    worked: bool

@app.post("/api/feedback")                             # the long-run adaptation loop
async def feedback(f: Feedback):
    ok = await remember(PLAYBOOK, f"{DEAL['company']} ({'WON' if f.worked else 'LOST'}): tactic '{f.tactic}' "
                  f"{'worked' if f.worked else 'did not work'} with the CFO.", context="tactic outcome")
    return {"ok": ok}

@app.get("/api/memories")
async def memories():
    try:
        r = await hs.alist_memories(bank_id=bank(), limit=30)
        items = getattr(r, "items", None) or getattr(r, "memories", None) or []
        return {"items": [{"type": getattr(i, "fact_type", ""), "text": getattr(i, "text", str(i))} for i in items]}
    except Exception as e: return {"items": [], "error": str(e)}
