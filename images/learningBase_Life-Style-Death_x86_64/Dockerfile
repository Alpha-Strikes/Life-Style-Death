FROM busybox

WORKDIR /tmp
RUN mkdir -p /tmp/learningBase/train /tmp/learningBase/validation

COPY data/training_data.csv /tmp/learningBase/train/training_data.csv
COPY data/test_data.csv      /tmp/learningBase/validation/test_data.csv

COPY images/learningBase_Life-Style-Death_x86_64/README.md /README.md