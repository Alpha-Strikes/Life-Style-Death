FROM busybox

WORKDIR /tmp
RUN mkdir -p /tmp/knowledgeBase

COPY models/currentAiSolution.h5   /tmp/knowledgeBase/currentAiSolution.h5
COPY models/currentOlsSolution.pkl /tmp/knowledgeBase/currentOlsSolution.pkl

COPY images/knowledgeBase_Life-Style-Death_x86_64/README.md /README.md