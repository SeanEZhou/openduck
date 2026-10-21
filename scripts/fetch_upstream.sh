#!/usr/bin/env bash
# Clone the upstream Open Duck repos at the commits pinned in configs/robot.yaml.
# Needed only for the upstream baseline policies (BEST_WALK_ONNX*.onnx) and for reference;
# everything the simulator needs is already vendored in assets/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party

clone() {  # repo commit
  local dir="third_party/$(basename "$1")"
  if [ ! -d "$dir" ]; then
    git clone -q "https://github.com/$1" "$dir"
  fi
  git -C "$dir" checkout -q "$2"
  echo "$1 @ $2"
}

clone apirrone/Open_Duck_Mini b23317a485b3cec7d8417f352478778b3475173c
clone apirrone/Open_Duck_Playground b9be205ac64488c23504ca42e5ec790337adeec3
clone apirrone/Open_Duck_Mini_Runtime 376de65435c93486347cd601a31aa96951799896
