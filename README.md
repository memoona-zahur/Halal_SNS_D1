# Halal SNS Moderation Checker

Automatic content moderation for the Halal SNS platform. The backend sends a post (a caption and image URLs). The checker checks the post and answers the same request, normally within a second or two, with whether to **allow**, **hold** or **block** it, a category and a one-line reason. If a check is slower than 10 s, the answer is `202` and the decision is read later.

**Status:** D1 (general moderation on upload) is implemented and tested. See [Project status](#project-status).

## Contents

1. [How it works](#how-it-works)
2. [What happens when you call the endpoint](#what-happens-when-you-call-the-endpoint)
3. [Repository layout](#repository-layout)
4. [Requirements](#requirements)
5. [Quick start](#quick-start)
6. [GPU or CPU](#gpu-or-cpu)
7. [Team testing over ngrok](#team-testing-over-ngrok)
8. [Configuration](#configuration)
9. [API](#api)
10. [Testing](#testing)
11. [Performance](#performance)
12. [Documentation](#documentation)
13. [Project status](#project-status)
14. [Contributing](#contributing)

## How it works

```mermaid
sequenceDiagram
    participant B as Backend
    participant A as Check door (API)
    participant D as Postgres
    participant W as Worker (models)
    B->>A: POST /v1/moderations (caption, image URLs)
    A->>D: save job, wake worker
    W->>D: claim job
    W->>W: check caption, images, text in images
    W->>D: save decision
    A->>D: check the job every 0.1 s (up to 10 s)
    A-->>B: 200 {status, label, category, reason}
    Note over B,A: Only if slower than 10 s: 202 {job_id}
    B->>A: GET /v1/moderations/{job_id} (after a 202)
    W-->>B: callback (optional, only if callback_url was sent)
```

| Check | Model | Runs on |
|---|---|---|
| Caption, and text found inside images | Qwen3Guard-Gen-0.6B | GPU or CPU |
| Each image | SigLIP2 (base, 384) plus a zero-shot head | GPU or CPU |
| Text inside images | RapidOCR (PP-OCRv5, ONNX) | CPU |

- The worst result across the caption and all images decides the post: `block` > `hold` > `allow`.
- Categories are G-01 to G-07: violence, weapons, sexual content, hate and harassment, self-harm, drugs and illegal goods, crime and fraud.
- An image that cannot be checked (too large, wrong type, unreadable, blocked host) is held and listed in the callback as `rejected_images`. The `200` and `GET` answers only say `hold` with a reason. The other images are still checked.
- Jobs, decisions and callbacks live in Postgres. A crash delays a post but never loses it or publishes it unchecked. A job's lease is renewed while it runs, and a crashed worker's job is picked up again.

## What happens when you call the endpoint

1. The API checks the body (types, unknown fields, text or images present, unique `media_id`), the surface (`400 unknown_surface` unless `"post"`) and the limits (`422 limit_exceeded`).
2. One database statement saves the job as `queued`. The same `surface` + `content_id` + `content_version` returns the existing job. The worker is woken, and the API waits for the decision: it checks the job every 0.1 s for up to `API_WAIT_S` (10 s).
3. The worker claims the oldest queued job and keeps it with a 20 s lease that it renews while it works. A crashed worker's job is picked up again.
4. The caption is cleaned (full-width letters and invisible characters undone) and goes to the text model, as a whole and as each paragraph, sentence and comma clause, so text pasted after a threat cannot talk the model out of it. One sentence on its own can hold a post, not block it. Personal details alone (an ID number, address or phone) are allowed and logged; if one clause of such a post is Unsafe (for example "share it so people harass her"), the post is blocked. Meanwhile every image is downloaded at the same time (https, allowed host, size, file type). A bad image is held and listed in `rejected_images`, and the others are still checked. An image that is not there yet retries the job.
5. Each image is scored by the image model and read by OCR. Text found in images is checked like a caption. If fewer than 3 letters are read, there is no text and the image is not held for it (a pattern or texture is not text). Text that is read at 3 or more letters but with low confidence is held as unreadable.
6. The rules turn each piece into allow, hold or block. The worst one wins, with a one-line reason.
7. One database statement saves the decision and marks the job done.
8. The API answers `200` with the decision (or with the failure), or `202 {job_id, status}` if the check is slower than the wait. After a `202` the backend polls `GET /v1/moderations/{job_id}` or waits for the callback. If `callback_url` was sent, the callback is also POSTed (optional, 5 s limit), and retried after 1, 5, 30, 120 and 600 s, up to 10 tries.

**When something goes wrong.** A temporary problem retries the job after 5, 15, 60 and 300 s, then fails it with an `error_code`. A permanent problem answers `200` with `status: failed` (and sends a `moderation.failed` callback if one was asked for). A worker crash requeues the job after the lease ends. If the database is down, the API answers `503` within 5 s.

## Repository layout

```
moderation/            the service: the only code that ships in the Docker images
  api/                   the Check door (app.py, routes.py, schemas.py)
  worker.py              loads the models, runs jobs, retries, sends callbacks
  moderate.py            checks one post; the worst result wins
  policy.py              the rules in Python: surfaces, thresholds, worst wins
  text_clean.py          caption cleaning and the paragraphs, sentences and clauses scored on their own
  fetch.py               image download: allowed hosts, size, file type
  store.py               Postgres access: jobs, decisions, callbacks to send
  prompts.py             every prompt: the image categories (SigLIP2) and the Qwen3Guard detail text
  domain.py              shared types: labels, categories, the request
  errors.py              the two error kinds
  logging_setup.py       JSON logs; caption and OCR text never reach them
  config.py              every setting, and the pinned model versions
  classifiers/           text (Qwen3Guard), image (SigLIP2), OCR (RapidOCR)
migrations/            database schema (001_init.sql)
tests/                 unit/ (no GPU, no database), integration/ (Postgres), models/ (real weights);
                       eval_text.py: runs the labelled captions in tests/data through the real text pipeline
testui/                local test page for the Check door
Dockerfile, docker-compose.gpu.yml, docker-compose.cpu.yml, pyproject.toml, .env.example
models/                model files: created by the download step, not in Git
model_download.py      download the models and build the image head (run it once on a new machine)
```

## Requirements

| | Needed for |
|---|---|
| Python 3.14 | the virtual environment: downloading the models, running the tests |
| Docker Desktop (Compose v2.24 or newer) | running Postgres, the API and the worker |
| About 3 GB of disk | the models, plus the Docker images (see [GPU or CPU](#gpu-or-cpu)) |
| An NVIDIA GPU | optional: only for the `gpu` worker |

## Quick start

Do the steps in order, from the project folder. Each step says what it does and how to know it worked. The commands are PowerShell (Windows); on Linux or macOS use `source .venv/bin/activate` and `/` in paths.

**Step 1. Create the Python environment and install the packages.** This environment is only for downloading the models and running the tests; the Docker images bring their own Python.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Done when `pip` finishes without an error. The CPU build of PyTorch is used here on purpose: it is the smaller download and is enough for the model download and the tests. [requirements.txt](requirements.txt) lists every package with its tested version.

**Step 2. Create your settings file.**

```powershell
copy .env.example .env
```

Open `.env` and set `IMAGE_ALLOWED_HOSTS`:

| Setting | Value |
|---|---|
| `IMAGE_ALLOWED_HOSTS` | the host your image URLs use, for example `res.cloudinary.com`. Empty refuses every image URL |

**Step 3. Download the models** (once, about 3 GB, needs internet).

```powershell
python model_download.py pull
```

Done when the `models\` folder exists and the command ends without an error. The worker cannot start without it.

**Step 4. Start the system** (Postgres, database setup, API and worker). Docker Desktop must be running. Pick the file for your machine: `docker-compose.gpu.yml` if it has an NVIDIA GPU, `docker-compose.cpu.yml` if not (see [GPU or CPU](#gpu-or-cpu)). The first build takes several minutes because it downloads PyTorch.

```powershell
docker compose -f docker-compose.gpu.yml up -d --build     # NVIDIA GPU
docker compose -f docker-compose.cpu.yml up -d --build     # no GPU
```

**Step 5. Check it works.**

```powershell
Invoke-RestMethod http://localhost:8000/readyz
docker compose -f docker-compose.gpu.yml logs worker      # or docker-compose.cpu.yml
```

Done when `readyz` answers `status: ready` and the worker log has a line with `"msg": "models ready"`. The API reference is at <http://localhost:8000/docs>.

**Step 6. Send a first post.**

```powershell
$body = @{ request_id = [guid]::NewGuid().ToString(); surface = "post"; content_id = "demo-1"; content_version = 1; text = "hello, nice weather today"; images = @() } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/v1/moderations -ContentType "application/json" -Body $body
```

Done when the answer has `"status": "done"` and `"label": "allow"`. The request waits for the decision (usually under a second), so the answer comes straight back. The full request and response are in [API](#api).

**Stop the system** with `docker compose -f docker-compose.gpu.yml down` (or the `cpu` file). Add `-v` only if you also want to delete the database.

**Try it by hand with a page.** `python testui\server.py`, then open <http://localhost:8090>. Write a caption, add images from the examples, an upload or **any image URL (for example Cloudinary)**, and send. The page sends the post exactly as the backend would, then shows the request, the response, the live status and the callback, and explains any problem in plain words. For real URLs the host must be in `IMAGE_ALLOWED_HOSTS`. Setup and tips are in [testui/README.md](testui/README.md).

## GPU or CPU

The worker is the part that runs the models. There are two compose files, one per kind of machine. Each starts the whole stack (Postgres, database setup, API and worker); they differ only in the worker.

| | GPU | CPU |
|---|---|---|
| Compose file | `docker-compose.gpu.yml` | `docker-compose.cpu.yml` |
| Needs | NVIDIA GPU and driver, Docker with GPU support | nothing special |
| PyTorch in the image | CUDA build | CPU-only build |
| Image size | about 12 GB | about 2.2 GB |
| Speed (measured, laptop RTX 3050 4 GB) | caption 0.3 to 0.4 s, ten images 1.8 s | caption 1.0 to 1.2 s, ten images 4.5 s |
| Start-up (models load) | about 10 s | about 11 s |
| GPU memory | about 1.3 GB | none |

Both give the same decisions. The CPU is about 3 times slower. Use the CPU file when there is no GPU, or for a quick local trial.

**Choose one** and use its file in every `docker compose` command:

```powershell
docker compose -f docker-compose.gpu.yml up -d --build     # NVIDIA GPU
docker compose -f docker-compose.cpu.yml up -d --build     # no GPU
```

**GPU setup.**
- Windows: a recent NVIDIA driver and Docker Desktop with the WSL2 backend. Run `nvidia-smi` in PowerShell: if it lists your GPU, Docker Desktop can normally use it.
- Linux: the NVIDIA driver and the NVIDIA Container Toolkit.
- Run one GPU worker per GPU: each process holds its own copy of the models. More jobs at once come from `FAST_WORKERS`, not from more workers.

**See which one is running.** The worker logs a line when its models are ready:

```powershell
docker compose -f docker-compose.gpu.yml logs worker       # or docker-compose.cpu.yml
```

Look for `"msg": "models ready", "device": "cuda"` (GPU) or `"device": "cpu"`.

**Switch.** Stop the stack with the file you started it with, then start with the other. The database is kept:

```powershell
docker compose -f docker-compose.gpu.yml down
docker compose -f docker-compose.cpu.yml up -d --build
```

**The `DEVICE` setting.** It decides what the worker does inside the container: `auto` (default) uses the GPU if it works and otherwise falls back to the CPU, `cuda` refuses to start without a working GPU, `cpu` forces the CPU. The CPU file sets `DEVICE=cpu` for you.

**Problems**

| You see | Cause and fix |
|---|---|
| `could not select device driver "nvidia"` when starting | Docker cannot see a GPU. Fix the driver or WSL2 GPU support, or use `docker-compose.cpu.yml`. |
| The worker exits at start with `no model at models/...` | The models are not downloaded. Run `python model_download.py pull`. |
| Posts stay `queued` | The worker is not running or not ready yet. Check `docker compose -f <file> ps` and the worker logs. |
| GPU out of memory | One worker per GPU. Lower `FAST_WORKERS` or `MODEL_MAX_BATCH` in `.env`. |
| Slow posts on the CPU | Expected. See the table above; the checker renews a job's lease while it runs, so slow jobs are not run twice. |

## Team testing over ngrok

For a team to hit the checker from their own machines (scripts, not browsers) when port forwarding is not possible (for example behind CGNAT). One reserved free-tier ngrok domain, one API key per teammate.

**One-time setup**

1. Create a free [ngrok](https://ngrok.com) account, get an auth token (<https://dashboard.ngrok.com/get-started/your-authtoken>), and reserve a free static domain (<https://dashboard.ngrok.com/domains>).
2. In `.env`, set:
   - `NGROK_AUTHTOKEN` - the auth token from step 1.
   - `NGROK_DOMAIN` - the reserved domain, e.g. `your-name.ngrok-free.app`.
   - `API_KEYS` - one `name:key` pair per teammate, comma-separated, e.g. `alice:<random-string>,bob:<random-string>`. Generate each key with `python -c "import secrets; print(secrets.token_urlsafe(24))"`.
3. Start the GPU stack; it now also starts the `ngrok` service, which tunnels only to `api` (never to `worker` or `postgres`):

   ```powershell
   docker compose -f docker-compose.gpu.yml up -d --build
   ```
4. Give each teammate the domain and their own key. They call it like any other deployment, with the key as a header:

   ```powershell
   curl https://your-name.ngrok-free.app/v1/moderations `
     -H "Content-Type: application/json" -H "X-API-Key: <their key>" `
     -d '{\"request_id\":\"t1\",\"surface\":\"post\",\"content_id\":\"p1\",\"content_version\":1,\"text\":\"hello\",\"images\":[]}'
   ```

   A teammate who would rather click through a page than write curl or scripts can instead run the project's test UI on their own machine, pointed at the tunnel: see [Testing from another PC, over the ngrok tunnel](testui/README.md#testing-from-another-pc-over-the-ngrok-tunnel) in `testui/README.md`.

**What to know before a team hits it in parallel**

- **Auth is the whole perimeter.** With `API_KEYS` unset, the API takes any request with no key at all - set it before starting `ngrok`, not after.
- **Images are normally URLs.** (Raw uploads exist but are off unless `IMAGE_ALLOWED_ROOT` is set; see [Raw image uploads](#raw-image-uploads).) The body is small JSON; the worker fetches image bytes itself from `IMAGE_ALLOWED_HOSTS`. Add every host your teammates' test images live on, or fetches are refused. Nothing from the tunnel's free monthly data cap goes toward image bytes.
- **Not every answer is `200`.** With several teammates firing at once, some posts queue behind `FAST_WORKERS` (4) and miss the `API_WAIT_S` (10 s) window - test scripts must handle `202 {job_id}` and poll `GET /v1/moderations/{job_id}`, the same as any other caller. The measured worst case (ten images, GPU) is 1.8 s, well inside both the wait window and ngrok's free-tier 60 s request timeout.
- **One worker process total.** `docker-compose.gpu.yml` runs a single worker by design - two would each load their own copy of the models and could overrun the 4 GB card. Throughput comes from `FAST_WORKERS` and `MODEL_MAX_BATCH`, not from more worker replicas; do not raise `replicas` on the `worker` service.
- **This is still D1.** No rate limiting, no key rotation, no signed callbacks (see [Project status](#project-status)). Fine for a team test behind a private key; tear down the tunnel (`docker compose -f docker-compose.gpu.yml stop ngrok`) when the team is done with it.

## Configuration

All settings are environment variables, read from `.env`. [.env.example](.env.example) lists every key with its default and a short note. The ones you will touch:

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | none (required) | Postgres connection string. Compose sets it for the containers. |
| `DEVICE` | `auto` | `auto`, `cuda` (refuse to start without a GPU) or `cpu`. |
| `IMAGE_ALLOWED_HOSTS` | empty | Hosts images may come from. Empty refuses every URL. |
| `IMAGE_ALLOWED_ROOT` | empty | Folder that raw uploads (`POST /v1/moderations/uploads`) and local image paths may use. Empty disables both. In Docker it must be `/app/uploads`, the folder compose shares between `api` and `worker`. |
| `ALLOW_HTTP_LOCALHOST` | `false` | Plain http to localhost for local testing only. |
| `API_MAX_IMAGES` / `IMAGE_MAX_BYTES` | 10 / 10 MiB | Images per post, and size of each image. An image is refused as soon as its download says it is too big, before the body is read. |
| `API_MAX_BODY_BYTES` | 128 KiB | Largest request body. A bigger one is answered `413` before it is read. |
| `FAST_WORKERS` | 4 | Jobs a worker runs at the same time. |
| `LEASE_S` | 20 | Seconds before a crashed worker's job is picked up again. |
| `MAX_ATTEMPTS` | 5 | Tries per job before it fails. |
| `API_WAIT_S` / `API_WAIT_POLL_S` | 10 s / 0.1 s | How long `POST /v1/moderations` waits for the decision before answering `202` (`0`: never wait), and how often it checks the job. |
| `CALLBACK_TIMEOUT_S` / `CALLBACK_MAX_ATTEMPTS` | 5 s / 10 | Callback delivery. |
| `API_KEYS` | empty (auth disabled) | `name:key` pairs, comma-separated. Required whenever the API is reachable from outside this machine (see [Team testing over ngrok](#team-testing-over-ngrok)). |

Model versions are pinned in `moderation/config.py` (`MODEL_PINS`).

## API

Interactive documentation, with every field and response code: `{BASE_URL}/docs` and `{BASE_URL}/redoc` (the raw OpenAPI file is `{BASE_URL}/openapi.json`). This section is the guide for calling the checker from the backend.

**URL structure**

Every URL is `{BASE_URL}` plus a path. `{BASE_URL}` is the address where the checker's `api` service is reachable:

| Where | `{BASE_URL}` |
|---|---|
| Local (Docker on your machine) | `http://localhost:8000` |
| Deployed | the host and port the API is published on, for example `https://moderation.internal.example.com`. Ask whoever runs the deployment. |

| Method and full URL | What it does | Success answer |
|---|---|---|
| `POST {BASE_URL}/v1/moderations` | Send a post and wait for the decision. | `200` decision, or `202 {job_id, status}` if the check takes longer than `API_WAIT_S` |
| `GET {BASE_URL}/v1/moderations/{job_id}` | State of one job. `{job_id}` is the UUID from a `202` (or a `200`). | `200` same body as the `POST` decision |
| `POST {BASE_URL}/v1/moderations/uploads` | Upload one raw image instead of hosting it. Needs `IMAGE_ALLOWED_ROOT` (see [Raw image uploads](#raw-image-uploads)). | `201 {media_id, url}` |
| `GET {BASE_URL}/healthz` | The process is up. | `200 {"status": "ok"}` |
| `GET {BASE_URL}/readyz` | The database answers. | `200 {"status": "ready"}`, or `503` |

- Only these paths exist. There is no `/api` prefix and no trailing slash: `/v1/moderations`, not `/v1/moderations/`.
- Send and receive JSON. On `POST` set the header `Content-Type: application/json`. If `API_KEYS` is set (required whenever the checker is reachable from outside this machine, for example over the ngrok tunnel), also send `X-API-Key: <your key>`; a missing or wrong key is `401 unauthorized`. `/healthz` and `/readyz` never need a key.
- The `POST` can wait up to `API_WAIT_S` (10 s) for the answer, so set the backend's HTTP timeout above that, for example 15 s.

**Send a post: `POST {BASE_URL}/v1/moderations`**

Body fields. A field that is not listed is refused (`422`), so do not send extras.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `request_id` | string, 1 to 128 characters | yes | Your id for this call. Used for tracing only. A new value per call is fine. |
| `surface` | string, up to 32 characters | yes | What is checked. Only `"post"` is accepted today; anything else is `400 unknown_surface`. |
| `content_id` | string, 1 to 128 characters | yes | Your id for the post. |
| `content_version` | integer, 1 or more | yes | Starts at 1. An edit of the same post sends a higher number and is checked again. |
| `author_id` | string, up to 128 characters | no | Id of the author. Stored for tracing. |
| `text` | string, up to 10,000 characters | one of `text` or `images` | The caption. |
| `images` | list of up to 10 objects | one of `text` or `images` | The images to check. Each object has `media_id` and `url` (next rows). |
| `images[].media_id` | string, 1 to 128 characters | yes, per image | Your id for the image. Must be unique within the request. Comes back in `rejected_images` and in the reason. |
| `images[].url` | string, up to 2,048 characters | yes, per image | Where the checker downloads the image. `https`, on a host in `IMAGE_ALLOWED_HOSTS`. |
| `callback_url` | URL, up to 500 characters | no | Where the checker POSTs the decision. Leave it out to poll with `GET` instead. See "Receive the callback" below. |

At least one of `text` or `images` must be present, and `media_id` values must be unique. `callback_url` must be a full `http` or `https` URL that the **worker** can reach: from inside Docker, `localhost` is the worker container itself, so use a real host name (on Docker Desktop, `host.docker.internal` reaches your machine).

Example body:

```json
{
  "request_id": "6f1c0c8e-3d52-4d0e-8c56-0a1f0c3b9e11",
  "surface": "post",
  "content_id": "post_123",
  "content_version": 1,
  "author_id": "u_456",
  "text": "Family picnic at the lake",
  "images": [{ "media_id": "m1", "url": "https://res.cloudinary.com/demo/image/upload/m1.jpg" }],
  "callback_url": "https://backend.example.com/moderation-result"
}
```

Try it from a terminal (replace `{BASE_URL}`). A caption-only post needs no image host:

```bash
curl -X POST "{BASE_URL}/v1/moderations" \
  -H "Content-Type: application/json" \
  -d '{"request_id":"t-1","surface":"post","content_id":"post_123","content_version":1,"text":"Family picnic at the lake"}'
```

```powershell
$body = @{ request_id = "t-1"; surface = "post"; content_id = "post_123"; content_version = 1; text = "Family picnic at the lake" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "{BASE_URL}/v1/moderations" -ContentType "application/json" -Body $body
```

**Answer `200`: the decision.** The check finished within the wait (normally 0.2 to 0.5 s on GPU, 1 to 2 s on CPU):

```json
{
  "job_id": "5b0e0a2e-7a43-4c36-9a58-1f6f7c1c2d10",
  "status": "done",
  "label": "allow",
  "category": null,
  "reason": "Allowed",
  "error_code": null,
  "latency_ms": 312
}
```

Response fields (the same body comes from `GET /v1/moderations/{job_id}`):

| Field | Meaning |
|---|---|
| `job_id` | The job's UUID. Use it for `GET` and to match the callback. |
| `status` | `queued` (waiting), `running` (being checked), `done` (decided), `failed` (could not be checked) or `dead` (gave up after the retries). Stop polling at `done`, `failed` or `dead`. |
| `label` | `allow`, `hold` or `block`. Set once `status` is `done`, otherwise `null`. |
| `category` | Harm category of the worst item, `G-01` to `G-07` (see [How it works](#how-it-works)). `null` when allowed, or when a hold or block has no category. |
| `reason` | One line naming the item and category, for example `Image m2: file is too large`. `Allowed` when allowed. `null` until decided. |
| `error_code` | Why a `failed` or `dead` job could not be checked (for example `media_not_found`, `attempts_exhausted`). Otherwise `null`. |
| `latency_ms` | Time from accepted to decided, queue wait included. `null` until decided. |

`status` is `done` (the decision is in `label`, `category` and `reason`), or `failed` or `dead` (the post could not be checked, see `error_code`; do not publish and ask the user to try again).

**Answer `202`: still running.** The check took longer than the wait. Keep the post hidden, then read the result with `GET /v1/moderations/{job_id}` (same body as the `200`) or from the callback:

```json
{ "job_id": "5b0e0a2e-7a43-4c36-9a58-1f6f7c1c2d10", "status": "queued" }
```

**Poll a job: `GET {BASE_URL}/v1/moderations/{job_id}`**

Use it after a `202`. Call it every second or two until `status` is `done`, `failed` or `dead`. A `job_id` that is not a UUID is `422`; an unknown one is `404 not_found`.

```bash
curl "{BASE_URL}/v1/moderations/5b0e0a2e-7a43-4c36-9a58-1f6f7c1c2d10"
```

**Receive the callback**

If `callback_url` was sent, the worker also POSTs the result there, as well as the answer above. To receive it, the backend exposes an endpoint at that URL which:

- accepts `POST` with a JSON body (`Content-Type: application/json`) and answers any `2xx` within 5 s. Anything else, or no answer, counts as a failure and is retried after 1, 5, 30, 120 and 600 s, up to 10 tries.
- may get the same callback more than once, so match on `job_id` (or `content_id` + `content_version`) and ignore repeats.
- is reachable from the worker. The callback is not signed and carries no secret, so keep it on a private network or check the `job_id` against jobs you submitted.

It is the only place that lists `rejected_images`:

```json
{
  "event_type": "moderation.decided",
  "job_id": "...",
  "content_id": "post_123",
  "content_version": 1,
  "label": "hold",
  "category": null,
  "reason": "Image m2: file is too large",
  "rejected_images": [{ "media_id": "m2", "error_code": "media_too_large", "detail": "file is too large" }]
}
```

If a post cannot be checked at all, the callback is `moderation.failed`:

```json
{
  "event_type": "moderation.failed",
  "job_id": "...",
  "content_id": "post_123",
  "content_version": 1,
  "error_code": "media_not_found"
}
```

`error_code` is one of `media_not_found`, `media_too_large`, `media_unsupported`, `media_forbidden`, `media_undecodable`, `model_oom`, `attempts_exhausted` or `internal_error`.

### Raw image uploads

An alternative to hosting the image yourself. It is off unless `IMAGE_ALLOWED_ROOT` is set (in Docker: `IMAGE_ALLOWED_ROOT=/app/uploads` in `.env`).

1. `POST {BASE_URL}/v1/moderations/uploads` as `multipart/form-data` with one field, `file`, holding the image. Send `X-API-Key` if `API_KEYS` is set.
2. The answer is `201 {"media_id": "...", "url": "uploads/<name>.jpg"}`. The type is judged by file content (JPEG, PNG, WebP or GIF), and the size limit is `IMAGE_MAX_BYTES`.
3. Put that `url` in `images[].url` on the next `POST /v1/moderations`. It is a path under `IMAGE_ALLOWED_ROOT`, not an `https` link, so `IMAGE_ALLOWED_HOSTS` does not apply to it.

After a decision, the worker also keeps a copy of every uploaded image in `uploads/by_category/<label>/<category>/` (for example `hold/G-03_Sexual_content_and_nudity`, or `hold/unclassified`), so you can look through what was held or blocked. The original stays where it is. Images fetched from `https` links are not copied.

**Errors from the API itself**

| HTTP | `error` | When | What to do |
|---|---|---|---|
| 400 | `unknown_surface` | `surface` is not `"post"` | Fix the request. |
| 404 | `not_found` | `GET` with a `job_id` that does not exist | Check the id. |
| 413 | `body_too_large` | Body larger than 128 KiB (an upload: larger than `IMAGE_MAX_BYTES` plus a small margin) | Send less. |
| 422 | `limit_exceeded` | Too many images, or text, image URL or callback URL too long. On an upload: the file is too big or not a JPEG, PNG, WebP or GIF | Stay within the limits below. |
| 422 | (none: a `detail` list) | Malformed body: wrong type, missing field, unknown field, no text or images, duplicate `media_id` | Fix the body. The list says which field. |
| 503 | `unavailable` | The database is down, or (on an upload) raw uploads are disabled because `IMAGE_ALLOWED_ROOT` is empty | Keep the post hidden and send the same request again, or host the image and send its link. |

Errors the API makes itself have the shape `{"error": "...", "detail": "..."}`. FastAPI's own `422` has `{"detail": [ ... ]}`.

**Integration steps for the backend**

1. A user submits a post. Save it as hidden (processing).
2. `POST {BASE_URL}/v1/moderations` with the caption, the image URLs and (optional) your `callback_url`. Use the same `content_id` and `content_version` if you have to send it again.
3. `200` with `status: done`: act on `label` (table below). `200` with `failed` or `dead`: do not publish; ask the user to retry.
4. `202`: keep it hidden and wait for the callback, or poll `GET`, until `done`, `failed` or `dead`.
5. `503`, a timeout or a network error: retry the same request later; it is safe.

| Label | What the backend does |
|---|---|
| `allow` | Publish. |
| `hold` | Keep hidden. |
| `block` | Do not publish. |

**Rules**

- Set the backend's HTTP timeout above `API_WAIT_S`, for example 15 s. Do not retry a timed-out call with a new `content_version`: send the same one again.
- Only `surface: "post"` is accepted today. Anything else returns `400 unknown_surface`.
- The same `surface` + `content_id` + `content_version` returns the same job. If it is already decided, the answer is `200` at once. An edit must send a higher `content_version`.
- Images are sent as links, not files: the post body has no file and no base64 field. If you would rather not host the image, upload it first (see [Raw image uploads](#raw-image-uploads)). Cloudinary is only an example. Any storage works (S3, Azure Blob, your own CDN or a backend route served over https), as long as the link is `https` on a host in `IMAGE_ALLOWED_HOSTS`. The host match is exact, with no wildcards, and a signed query string is fine.
- Image URLs must be `https` on a host in `IMAGE_ALLOWED_HOSTS`. Accepted types: JPEG, PNG, WebP and GIF, judged by file content. AVIF is not supported, so do not use Cloudinary's `f_auto`.
- Animated images (GIF, WebP, APNG) are held until video checks exist.
- Limits: 10 images of 10 MiB, caption 10,000 characters, image URL 2,048, callback URL 500.

## Testing

```powershell
ruff check moderation tests testui model_download.py
mypy
pytest tests/unit -q                       # no database or models needed
pytest tests/models -q                     # real models; needs the downloaded weights
```

The integration tests need Postgres and a separate database, because they empty every table. Once:

```powershell
docker compose -f docker-compose.cpu.yml up -d postgres migrate
docker compose -f docker-compose.cpu.yml exec -T postgres psql -U moderation -d moderation -c "CREATE DATABASE moderation_test"
```

Then, in each new terminal:

```powershell
$env:TEST_DATABASE_URL = "postgresql://moderation:moderation@localhost:5432/moderation_test"
pytest tests/integration -q
```

| Suite | Count | Needs |
|---|---|---|
| Unit | 204 | nothing |
| Integration | 21 | Postgres (uses a separate `moderation_test` database) |
| Model | 22 | the downloaded models |

The unit suite passed on 3 November 2026 (204 tests). The integration and model suites last passed on 28 October 2026 and were not rerun after the OCR rule change and the removal of the calibration tools. The manual and end-to-end cases, with real GPU and CPU results, are in the Test Cases document (see [Documentation](#documentation)).

## Performance

Measured on one laptop with an NVIDIA RTX 3050 (4 GB), 22 October 2026. Time from accepted to decided, queue wait included. This is also how long the `POST` takes to answer, so on GPU the answer is normally `200`.

| Post | GPU | CPU |
|---|---|---|
| Caption only | 0.3 to 0.4 s | 1.0 to 1.2 s |
| One image | 0.2 to 0.4 s | 0.6 to 1.6 s |
| Ten images | 1.8 s | 4.5 s |
| An image that is rejected | 11 to 16 ms | 13 to 41 ms |

Start-up loads three models: about 10 s on GPU and about 11 s on CPU. The GPU worker uses about 1.3 GB of GPU memory. Server numbers will differ.

## Documentation

The handover documents are kept outside this repository and are shared separately:

| Document | For |
|---|---|
| Backend Integration Guide | Backend developers: what to call and what comes back |
| Plan and handover | The design, what is built, open issues, next steps (section 16) |
| Request flow diagram, module map | One-page pictures of a post's path and of the code modules |
| Test cases | Every test case with its GPU and CPU result |

The live API reference is always at <http://localhost:8000/docs>. `testui/make_test_doc.py` regenerates the Test Cases document from `testui/cases.json` and `testui/results.json`.

## Project status

| Deliverable | Scope | State |
|---|---|---|
| D1 | General moderation on upload (posts: caption and images) | Implemented and tested |
| D2 | Video | Not started |
| D3 | Halal categories, other content types, calibrated thresholds, human review | Not started |
| D4 | Chat review | Not started |
| D5 | Recommendations | Not started |

**Known limits of D1** (full list in section 16.4 of the handover):

- Callbacks are not signed and carry no secret. The API itself takes a per-teammate key via `X-API-Key` when `API_KEYS` is set, but there is still no rate limiting, request logging beyond the access log, or key rotation - fine for a team testing over the ngrok tunnel in [Team testing over ngrok](#team-testing-over-ngrok), not a substitute for real auth in production.
- The picture thresholds were set per category from a scan of 347 real pictures (the calibration scripts have since been removed; the data is archived in `local/backup/samples-2026-11-03.zip`): clean pictures wrongly flagged are 3%, but weapons in pictures are mostly missed. A labelled test set, a trained image classifier and full calibration are part of D3.
- The text model has no weapon, drug or scam category. Detail text in its prompt (`CATEGORY_DETAILS` in `moderation/prompts.py`) makes it report those as Violent (G-01) or Non-violent Illegal Acts (G-07), never G-02 or G-06. There are no wording rules: the model alone decides, so a scam phrased as "verify your account password" is allowed. Threats written in Roman Urdu are missed. Check changes with `python tests/eval_text.py` (labelled cases in `tests/data/text_cases.jsonl`).
- Posts that only share personal details (ID number, address, phone) are allowed, even without a reason given. The post is blocked only when a clause of it is Unsafe (a threat or a call to harass). The model separates these only clause by clause, so a long post with the intent buried in a long clause can be missed.
- Scoring each sentence and clause costs time: about 0.3 s for a short caption and 0.7 s for five sentences on a GPU, and seconds on a CPU.
- The models that read text in images cover Chinese, English and Japanese. Text in Arabic or Urdu is not read: if the models read fewer than 3 letters from it, the image is judged by the image model alone; if they read garbled letters, it is held as unreadable.
- A post whose image URL stays dead fails after the retries, instead of being held.

## Contributing

- Work on a branch and open a pull request into `main`.
- Before you push, run ruff, mypy and `pytest tests/unit` (see [Testing](#testing)) and keep them clean.
- Never commit `.env` or model files. They are git-ignored.
- Change `D1_POLICY.version` in `moderation/policy.py` whenever you change a rule: it is saved with every decision.
- After editing `IMAGE_PROMPTS` in `moderation/prompts.py`, run `python model_download.py build-head`.

The models are third-party: Qwen3Guard, SigLIP2 and RapidOCR each have their own licence. Check them before any public release.
