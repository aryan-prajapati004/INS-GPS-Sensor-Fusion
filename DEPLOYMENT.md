# Deployment guide — local first, then Render

This walks through the whole path: get your trained weights in, test
everything on your own machine, then deploy to Render, then make the
handful of production tweaks you only need once it's live.

---

## Phase 0 — Prerequisites

- Python 3.11+ installed locally
- Your trained notebook has been run to completion, so
  `fusion_outputs/saved_models/*.pkl` exists (12 files)
- A GitHub account (Render deploys from a git repo)
- A Render account (free tier is enough) — sign up at render.com
- (Optional but recommended) Docker Desktop, to test the exact container
  Render will run before you push

---

## Phase 1 — Get your trained artifacts into the project

```bash
# from wherever you unzipped the project
cp /path/to/fusion_outputs/saved_models/*.pkl backend/artifacts/
ls backend/artifacts/
```

You should see exactly these 12 files:
```
RNN.pkl  LSTM.pkl  GRU.pkl  BiRNN.pkl  BiLSTM.pkl  Transformer.pkl
BERT.pkl  TFT.pkl  S4.pkl  Mamba.pkl  Mamba2.pkl  BiMamba.pkl
```
If a name doesn't match exactly (case-sensitive), the backend will report
that model as "no artifact" instead of crashing — but fix the name so you
get all 12.

---

## Phase 2 — Test the backend locally (no Docker yet)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

uvicorn main:app --reload --port 8000
```

Now check it's actually working:

```bash
curl http://localhost:8000/health
# {"status":"ok","loaded_models":["RNN","LSTM",...]}   <- all 12 should be here
```

If `loaded_models` is missing some names, check the log output for
`"Artifact missing for '<name>'"` — that tells you exactly which `.pkl`
didn't load and why.

Also open **`http://localhost:8000/docs`** in a browser — FastAPI's
auto-generated interactive docs. You can literally click "Try it out" on
`/predict/{model_name}` and `/leaderboard` without writing any code, which
is the fastest way to sanity-check real numbers coming out of your real
trained weights.

**Leave this running** — you need it live for the next phase.

---

## Phase 3 — Test the dashboard locally, against the local backend

Open `frontend/config.js` and point it at your local server:

```js
window.API_BASE_URL = "http://localhost:8000";
```

Then serve the frontend (in a second terminal, backend still running):

```bash
cd frontend
python3 -m http.server 5500
```

Open `http://localhost:5500` in a browser. You should see:
- The status dot go cyan ("online — 12/12 models ready")
- The leaderboard populate with your real ADE numbers, sorted best-first
- Clicking model chips toggles them (color-coded)
- **"Run Synthetic Field Path"** — select 2-3 models, click it, and you
  should see predicted trajectories overlaid against the synthetic ground
  truth on the scope. This is the same code path `/predict_trajectory`
  uses in production, so if it works here, it'll work on Render.

If the leaderboard shows "no artifact" for any model, go back to Phase 1.
If you get a red status dot / "backend unreachable", check the backend
terminal is still running and the URL in `config.js` matches its port.

---

## Phase 4 — Test the actual Docker image (recommended before pushing)

This catches "works on my machine but not in the container" issues before
Render does — e.g. a dependency your local venv had installed some other
way, or a path that only exists locally.

```bash
cd backend
docker build -t ins-gps-api .
docker run -p 8000:8000 ins-gps-api
```

Same health check as Phase 2:
```bash
curl http://localhost:8000/health
```

If this works, point `frontend/config.js` back at
`http://localhost:8000` and re-run the Phase 3 browser check once more
against the containerized version. This is the closest possible
simulation of Render's environment.

---

## Phase 5 — Push to GitHub

```bash
cd /path/to/project-root       # the folder containing backend/, frontend/, render.yaml
git init
git add .
git commit -m "INS/GPS fusion deployment"
git branch -M main
git remote add origin https://github.com/<you>/ins-gps-fusion.git
git push -u origin main
```

Double check `backend/artifacts/*.pkl` actually got committed (`git log
--stat` or `git show --stat HEAD`) — a `.gitignore` copied from another
project sometimes excludes `.pkl` by accident.

---

## Phase 6 — Deploy to Render

**Option A — Blueprint (one click, recommended):**
1. Render dashboard → **New → Blueprint**
2. Connect the GitHub repo you just pushed
3. Render reads `render.yaml` and proposes both services — review and
   click **Apply**
4. Wait for both builds to finish (backend build takes a few minutes —
   it's downloading the CPU torch wheel)

**Option B — Manual, if you want more control over each service:**
1. **New → Web Service** → connect repo → set **Root Directory** to
   `backend` → Runtime: **Docker** → Plan: **Free** → Create
2. **New → Static Site** → connect repo → set **Root Directory** to
   `frontend` → Publish directory: `.` → Plan: **Free** → Create

Either way, once the backend service is live, copy its URL — it'll look
like `https://ins-gps-fusion-api-xxxx.onrender.com`.

---

## Phase 7 — The "needed modifications" once it's live

These are the things that only make sense to change *after* you have a
real URL:

**1. Point the dashboard at the real backend.**
Edit `frontend/config.js`:
```js
window.API_BASE_URL = "https://ins-gps-fusion-api-xxxx.onrender.com";
```
Commit and push — the static site auto-redeploys. (Or edit it directly in
the Render dashboard's file editor if you enabled that, for a faster
loop.)

**2. Tighten CORS.**
In `backend/main.py`, change:
```python
allow_origins=["*"]
```
to your actual dashboard URL:
```python
allow_origins=["https://ins-gps-fusion-dashboard-xxxx.onrender.com"]
```
Commit, push, wait for the backend to redeploy.

**3. Verify the cold-start behavior.**
Free-tier services sleep after ~15 min idle. Hit `/health` after a period
of inactivity and time it — expect 30-60 seconds for the container to
spin up and reload all 12 models. If that's too slow for your use case,
Render's paid "Starter" tier keeps it always-on.

**4. Sanity-check the real leaderboard numbers.**
Open the deployed dashboard and confirm the ADE values match what your
notebook printed at training time — this is really just re-running Phase
3's browser check, but against the live URL, as a final confirmation that
nothing got corrupted in transit (e.g. a `.pkl` that got Git LFS'd
incorrectly, or an artifact that didn't survive the git push).

---

## Troubleshooting quick-reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `/health` shows fewer than 12 `loaded_models` | `.pkl` filename mismatch or missing file | Check exact names in `backend/artifacts/`, case-sensitive |
| Dashboard shows red "backend unreachable" | Wrong `API_BASE_URL`, or backend still cold-starting | Check `config.js`, wait 60s, retry |
| CORS error in browser console | `allow_origins` doesn't include the dashboard's actual URL | Update Phase 7 step 2 |
| Docker build fails on torch install | Network hiccup pulling the CPU wheel | Re-run build; check the `--index-url` line in the Dockerfile is intact |
| Predictions look wrong / all near zero | Scaler mismatch — `f_scaler`/`t_scaler` from a different training run than the `state_dict` in the same `.pkl` | Re-export from the notebook; don't mix-and-match artifact files |
| Render free-tier build fails on disk space | Accidentally installed the CUDA torch wheel instead of CPU | Confirm the Dockerfile's `--index-url https://download.pytorch.org/whl/cpu` line wasn't edited out |
