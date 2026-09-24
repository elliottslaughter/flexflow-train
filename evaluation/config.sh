# Which YOLOv10 to run and how. Sourced by the other scripts, which expect $HERE
# to be set. Each setting can be overridden from the environment, e.g.
#
#   MODEL=yolov10l BATCH=8 FSIZE=27000 ./benchmark_flexflow.sh
#
# MODEL   yolov10n, yolov10s, yolov10m, yolov10b, yolov10l or yolov10x
# BATCH   batch size
# FSIZE   size in MB of the pool Realm allocates up front on the GPU. It has to
#         hold every tensor of the model, and whatever it leaves of the GPU is
#         all that CUDA, cuDNN and the convolution workspace get. See README.md
#         for the values that fit each configuration.
#
# Sets CONFIG, which names the configuration in reports and work directories,
# and exports YOLOV10_MODEL, which tells the Python side which model to build
# (see yolov10-validation/reference_model.py).

MODEL="${MODEL:-yolov10x}"
BATCH="${BATCH:-6}"
FSIZE="${FSIZE:-27000}"

CONFIG="$MODEL-batch$BATCH"

export YOLOV10_MODEL="$MODEL"
