---
title: MedBrief AI
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# MedBrief AI

MedBrief AI is a multi-domain assistant that can still run locally, but now also supports a public deployment path. The FastAPI app serves the website, can call either a local model runtime or a hosted OpenAI-compatible provider, fetches public websites for grounding when medical or factual research is needed, cites the sources it used, and can persist authenticated user conversations in Supabase.

## What It Does

- serves a polished premium chat website from FastAPI
- uses a hosted OpenAI-compatible provider in production, or a local runtime such as Ollama in local development
- performs live web research for medical and factual questions without paid APIs or hosted AI endpoints
- prioritizes trusted public medical sources such as NIH, CDC, MedlinePlus, Mayo Clinic, NHS, WHO, Merck Manuals, and similar sites
- handles psychology-aware support, project explanation, portfolio questions, healthcare information, and general structured help
- supports account-backed sessions for public deployment
- stores conversations, profile settings, and memory either locally or in Supabase
- streams research phases such as searching, reading, comparing, and writing
- stays non-diagnostic and non-prescribing for health content

## Project Layout

```text
serve.py
runtime.py
grounding.py
web_research.py
presets.py

server/
  app.py
  auth.py
  hosted_provider_client.py
  ollama_client.py
  rate_limit.py
  schemas.py
  service.py
  settings.py
  storage.py
  supabase_client.py
  supabase_storage.py

frontend_static/
  index.html
  styles.css
  app.js
  engine.js
  config.js

Dockerfile
supabase/schema.sql
data/knowledge/
data/index/
data/app/
checkpoints/
```

## Setup

Use Python 3.12 if possible:

```powershell
py -3.12 -m venv .venv312
.venv312\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## Run The Website

The easiest command is:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_web_demo.ps1
```

Then open:

```text
http://127.0.0.1:8000
```

If Supabase is not configured locally, the app automatically falls back to a local development session so you can keep building without a public auth backend.

For local development:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_web_dev.ps1
```

That runs FastAPI on `http://127.0.0.1:8000` and serves the themed static frontend from `frontend_static`.

## Default Behavior

- product name: `MedBrief AI`
- first-party website: same-origin only
- default preset: `medbrief-medical`
- response mode: `assistant`
- live web research: enabled
- local knowledge retrieval: enabled
- public account auth: enabled when Supabase credentials are configured
- personalized preferences: enabled

## Environment

The server reads `.env`. The starter values in `.env.example` already match MedBrief AI defaults.

Key variables:

- `LLM_PRODUCT_NAME`
- `LLM_MODEL_BACKEND`
- `LLM_PROVIDER_BASE_URL`
- `LLM_PROVIDER_API_KEY`
- `LLM_PROVIDER_MODEL`
- `LLM_OLLAMA_BASE_URL`
- `LLM_OLLAMA_MODEL`
- `LLM_SESSION_SECRET`
- `LLM_KNOWLEDGE_INDEX`
- `LLM_DATABASE_PATH`
- `LLM_WEB_SEARCH_ENABLED`
- `LLM_WEB_SEARCH_MAX_RESULTS`
- `LLM_GENERATION_TIMEOUT_SECONDS`
- `LLM_MESSAGE_RATE_LIMIT`
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`

## Public Deployment

For a zero-budget public deployment:

1. Run the SQL in `supabase/schema.sql` inside your Supabase project.
2. In Supabase, create a hosted project, then copy the project URL, the legacy `anon` key, and the legacy `service_role` key into your deploy environment.
3. In Supabase Auth, keep email/password auth enabled and decide whether you want email confirmation on. This code signs users in immediately after signup, so the easiest v1 launch is to disable email confirmation until you also add SMTP and redirect pages.
4. In Supabase Auth URL settings, set the Site URL to your real production domain. If you later add password reset or email confirmation flows, add every allowed redirect URL there too.
5. Set `LLM_MODEL_BACKEND=hosted_api`.
6. Set `LLM_PROVIDER_BASE_URL`, `LLM_PROVIDER_API_KEY`, and `LLM_PROVIDER_MODEL` for your hosted OpenAI-compatible provider.
7. Set `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, and a strong `LLM_SESSION_SECRET`.
8. Set `LLM_ALLOW_ORIGINS` to your deployed HTTPS domain.
9. Deploy the repo with the included `Dockerfile` on a Docker-compatible host such as Hugging Face Spaces.
10. After the site is live, go back through the Supabase production checklist: stronger auth protections, SMTP if you keep email flows, and auth rate-limit review.

The website stays same-origin, the backend keeps the provider key on the server, and end users never talk directly to Supabase or the model provider from the browser.

For Hugging Face Spaces specifically:

1. Create a new Space and choose the `Docker` SDK.
2. Push this repository to that Space.
3. Add the same environment variables from `.env.example` as Space secrets.
4. Keep the app listening on port `7860` inside the container. The included `Dockerfile` already does this.
5. Use the default `*.hf.space` URL for a free launch. If you want a custom branded domain, budget for a domain purchase and a hosting plan that supports custom domains for your setup.

## Building The Local Knowledge Index

```powershell
py retrieval.py build `
  --knowledge-dir data/knowledge `
  --output data/index/knowledge_index.pkl `
  --chunk-words 350 `
  --overlap-words 50
```

## Training

The original toy training stack is still present for experimentation, but production MedBrief answers should use the stronger local runtime path by default.

If you still want to train the small local checkpoint:

```powershell
py train.py `
  --input cleaned_data.txt `
  --fiction-input cleaned_data.txt `
  --general-knowledge-dir data/knowledge/general `
  --medical-knowledge-dir data/knowledge/medical `
  --seed-chat-dir data/chat_seed `
  --general-weight 7 `
  --medical-weight 2 `
  --fiction-weight 1 `
  --system-preset factual-medical-lite `
  --model-preset base `
  --tokenizer-prefix data/tokenizer/advanced_local `
  --output-dir checkpoints/advanced_local `
  --knowledge-index-path data/index/knowledge_index.pkl `
  --resume checkpoints/advanced_local
```

The blended trainer can still run even if `cleaned_data.txt` is missing, because it rebuilds from `data/knowledge/*` and `data/chat_seed/*`. The `cleaned_data.txt` arguments are legacy placeholders unless you intentionally add that file back.

## Safety Behavior

MedBrief AI is designed for psychology-aware support, careful healthcare information, polished portfolio explanation, and strong general help. It is not a replacement for therapy, prescribing, diagnosis, or emergency care.

- emergency or red-flag symptom prompts should trigger urgent-care guidance
- medication dosing and prescribing requests should be declined safely
- diagnosis-style prompts should be answered cautiously and non-diagnostically
- weak or conflicting evidence should produce uncertainty language instead of fabricated certainty

## Publishing Notes

- keep the first-party website on same-origin routes only
- do not expose the internal compatibility routes in the public UI
- keep hosted provider keys and Supabase service keys on the server only
- keep Supabase RLS enabled on the app tables so browser-side anon traffic cannot query them directly
- verify your allowed origins before exposing the site publicly
- use HTTPS before internet-facing deployment
- keep `LLM_API_KEY_SELF_SERVE_ENABLED=false` unless you intentionally want public developer keys

## Internal Compatibility Routes

The server can still expose internal JSON routes such as `/v1/chat/completions` for controlled integrations, but the website itself does not depend on them and should never ask end users for credentials.

## Current Limits

- answer quality still depends on the upstream hosted model or your local runtime
- live web research improves breadth, but first-time answers can still be slower than cached repeats
- this is a medical information assistant, not a clinician
- free-tier public hosting can still have cold starts, limited throughput, and sleep behavior
