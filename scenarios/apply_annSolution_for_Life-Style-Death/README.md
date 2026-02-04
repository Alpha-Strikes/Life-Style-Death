# Apply ANN solution for Life-Style-Death

Run the Life-Style-Death ANN (and optionally OLS) inference via Docker Compose.

## Prerequisite: create the shared volume

All compose files use an **external** volume `ai_system`. Create it once:

```bash
docker volume create ai_system
```

## Variants

| Directory   | Use case |
|------------|----------|
| **x86_64** | Standard CPU (Linux/Windows x86_64). |
| **x86_64_gpu** | NVIDIA GPU. Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Uses the same `codebase_life_style_death_x86_64` image with `runtime: nvidia`. |
| **aarch64** | ARM64 (e.g. Apple Silicon). Requires image `codebase_life_style_death_aarch64`; see **Building the aarch64 image** below. |

## Run (example: x86_64)

```bash
cd scenarios/apply_annSolution_for_Life-Style-Death/x86_64
docker compose up --abort-on-container-exit
```

Or with classic Compose:

```bash
docker-compose up --abort-on-container-exit
```

Expected output ends with a line like: `Predicted age at death: XX.XX`.

## Images (Docker Hub)

- `moazemad/learningbase_life_style_death_x86_64`
- `moazemad/knowledgebase_life_style_death_x86_64`
- `moazemad/activationbase_life_style_death_x86_64`
- `moazemad/codebase_life_style_death_x86_64` (x86_64 and x86_64_gpu)
- `moazemad/codebase_life_style_death_aarch64` (for aarch64 scenario; build from repo — see below)

### Building the aarch64 image

From the **repository root**, on Apple Silicon (or any ARM64 host):

```bash
docker build -f images/codeBase_Life-Style-Death_aarch64/Dockerfile -t moazemad/codebase_life_style_death_aarch64 .
docker push moazemad/codebase_life_style_death_aarch64
```

From an x86_64 host (cross-build):

```bash
docker buildx build --platform linux/arm64 -f images/codeBase_Life-Style-Death_aarch64/Dockerfile -t moazemad/codebase_life_style_death_aarch64 --load .
docker push moazemad/codebase_life_style_death_aarch64
```
