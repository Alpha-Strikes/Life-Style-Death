# syntax=docker/dockerfile:1
FROM busybox
RUN mkdir -p /knowledgeBase
COPY models/currentAiSolution.h5   /knowledgeBase/currentAiSolution.h5
COPY models/currentOlsSolution.pkl /knowledgeBase/currentOlsSolution.pkl
COPY images/knowledgeBase_Life-Style-Death_x86_64/README.md /README.md