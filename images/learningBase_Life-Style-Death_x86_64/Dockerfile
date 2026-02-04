# syntax=docker/dockerfile:1
FROM busybox
#content under /learningBase (not /tmp) so that when ai_system is mounted at /tmp,
# a compose command can copy /learningBase -> /tmp/learningBase and expose it on the volume.
RUN mkdir -p /learningBase/train /learningBase/validation
COPY data/training_data.csv /learningBase/train/training_data.csv
COPY data/test_data.csv     /learningBase/validation/test_data.csv
COPY images/learningBase_Life-Style-Death_x86_64/README.md /README.md