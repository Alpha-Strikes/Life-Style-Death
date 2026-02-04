# codeBase_Life-Style-Death_aarch64

- **Owners**: Moaz Hussein, Nathan Fernandes
- **Course**: Created as part of “M. Grum: Advanced AI-based Application Systems” by the Junior Chair for Business Information Science, esp. AI-based Application Systems, University of Potsdam.
- **Content (code base)**:
  - Example activation dataset: `/tmp/acGvaGonBase/activation_data.csv`
- **Role**: ARM64 (aarch64) image; supplies the inference code to apply the ANN/OLS models from the knowledgeBase image. Same logic as the x86_64 codeBase, built for Linux/arm64 (e.g. Apple Silicon).
- **License**: The contents of this image are provided under the **AGPL-3.0** license.

## Build and push (from repository root)

On **Apple Silicon (aarch64)**:
```bash
docker build -f images/codeBase_Life-Style-Death_aarch64/Dockerfile -t moazemad/codebase_life_style_death_aarch64 .
docker push moazemad/codebase_life_style_death_aarch64
```

From **x86_64 host** (cross-build for arm64):
```bash
docker buildx build --platform linux/arm64 -f images/codeBase_Life-Style-Death_aarch64/Dockerfile -t moazemad/codebase_life_style_death_aarch64 --load .
docker push moazemad/codebase_life_style_death_aarch64
```
