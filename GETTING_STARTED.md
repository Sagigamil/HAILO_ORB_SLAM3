# Getting Started — Hailo-accelerated ORB-SLAM3

This fork offloads **FAST keypoint detection at pyramid level 0** from CPU to a
**Hailo-8** NPU. The Hailo HEF produces a per-pixel corner-weight heatmap; a 3×3
NMS over that heatmap replaces the cell-based `cv::FAST` scan upstream uses.
Pyramid levels 1..N-1 still run `cv::FAST` on CPU, and the descriptor pipeline
is unchanged.

If the HEF or device isn't available, the extractor transparently falls back to
the upstream `cv::FAST` path on every level, so SLAM still runs.

---

## 1. Prerequisites

Verified setup:
- Raspberry Pi 5, 64-bit Raspberry Pi OS
- Hailo-8 (PCIe HAT)
- HailoRT 4.23.0 (lib + driver + CLI)
- GCC 14, CMake ≥ 3.10
- OpenCV ≥ 4.4, Eigen3 ≥ 3.1, Pangolin

Check the Hailo install:

```bash
hailortcli fw-control identify     # should report your Hailo-8 device
dpkg -l | grep hailort             # 4.23.0 expected
ls /usr/lib/cmake/HailoRT/         # CMake config used by find_package(HailoRT)
ls /usr/lib/libhailort.so          # shared lib that gets linked
```

The remaining ORB-SLAM3 dependencies are documented in upstream `README.md` and
`Dependencies.md`.

---

## 2. Build

One-shot script (clones nothing — assumes you've already cloned this repo):

```bash
git clone https://github.com/Sagigamil/HAILO_ORB_SLAM3.git
cd HAILO_ORB_SLAM3
git checkout hailo-integration

chmod +x build.sh
./build.sh
```

`build.sh` builds the three Thirdparty libraries (DBoW2, g2o, Sophus),
unpacks the ORB vocabulary, then builds the main library and examples
under `build/`. Executables land under `Examples/*/`.

If you only want to rebuild the library + one example after a code change:

```bash
cmake --build build --target ORB_SLAM3 rgbd_tum -j$(nproc)
```

> **C++14 note.** The Thirdparty `CMakeLists.txt` files bump the standard to
> C++14 and drop `-Werror` in Sophus — required to compile under GCC 14.

---

## 3. Run the TUM RGB-D example

Grab the dataset (any TUM RGB-D sequence works; `fr1/xyz` is small and a good
first run):

```bash
mkdir -p ~/datasets/tum && cd ~/datasets/tum
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
tar xf rgbd_dataset_freiburg1_xyz.tgz
```

Run from the repo root:

```bash
cd /path/to/HAILO_ORB_SLAM3
./Examples/RGB-D/rgbd_tum \
    Vocabulary/ORBvoc.txt \
    Examples/RGB-D/TUM1.yaml \
    ~/datasets/tum/rgbd_dataset_freiburg1_xyz \
    Examples/RGB-D/associations/fr1_xyz.txt
```

At the end of the run you'll see:

```
Images in the sequence: 792
New Map created with 833 points
median tracking time: 0.038...
mean tracking time:   0.039...

[ORBextractor] level-0 path tally over 792 frames:
  hailo_ok       = 792
  fallback_init  = 0  (mHailoL0 null)
  fallback_run   = 0  (Run() returned false)
  fallback_shape = 0  (heatmap dims mismatch)
```

`hailo_ok = total frames` confirms every frame's level-0 keypoints came from
the NPU.

### GUI vs headless

`Examples/RGB-D/rgbd_tum.cc` controls the viewer via the 4th `System(...)` arg:

- `true`  → Pangolin window opens (needs a display; `DISPLAY=:0` on the Pi)
- `false` → headless; useful for benchmarking, no Qt segfault on shutdown

Rebuild with `cmake --build build --target rgbd_tum -j$(nproc)` after toggling.

---

## 4. Configuration

### `ORB_SLAM3_HAILO_L0_HEF`

Path to the HEF the extractor loads. Defaults to `Hailo/dense_fast_stage_L0.hef`
(checked in to this branch). Override if you have a custom build:

```bash
export ORB_SLAM3_HAILO_L0_HEF=/abs/path/to/your.hef
./Examples/RGB-D/rgbd_tum ...
```

Constraints on the HEF:
- Single input named `*/input_layer1`, UINT8, **480×640×1**
- One output that's UINT8 **NHWC 480×640×1** (the heatmap — recognised by name `activation1` or by matching the input dims)
- Other outputs are tolerated and ignored

### `HAILO_MONITOR`

Standard HailoRT runtime monitor. Setting `HAILO_MONITOR=1` makes HailoRT
publish a stats file under `/tmp/hmon_files/<pid>...`:

```bash
HAILO_MONITOR=1 ./Examples/RGB-D/rgbd_tum ...
```

Caveat: in HailoRT 4.23.0, `hailortcli monitor` does not render anything for
the async `InferModel` API we use, even though the file is being written. The
env var has no effect on the run itself beyond enabling that file.

### Threshold tuning

`ORBextractor.iniThFAST` / `minThFAST` in the YAML (`Examples/RGB-D/TUM1.yaml`,
etc.) are reused as **heatmap byte thresholds** for the NMS pass. The HEF
output range isn't identical to cv::FAST's score distribution, so if you swap
the model you may want to retune. With `iniThFAST=20` on this HEF, level 0
yields ~3k candidates per frame on TUM, which `DistributeOctTree` trims down
to ~170 keypoints.

---

## 5. What's actually offloaded

Files of interest:

| File | Role |
|---|---|
| `include/HailoFeatureExtractor.h`, `src/HailoFeatureExtractor.cc` | Thin C++ wrapper around HailoRT async-infer: loads HEF, runs sync inference on one 480×640 grayscale frame, exposes `activation1` as a `cv::Mat` view. |
| `src/ORBextractor.cc` — `ExtractKeypointsFromHailoHeatmap()` | 3×3 NMS scan over the heatmap inside `[minBorderX, maxBorderX) × [minBorderY, maxBorderY)`. Emits `cv::KeyPoint`s with `.response` = heatmap byte. |
| `src/ORBextractor.cc` — `ComputeKeyPointsOctTree()` | For `level == 0` runs the Hailo wrapper; on success skips the upstream cell-based `cv::FAST` loop. Levels 1..N-1 unchanged. |
| `Hailo/dense_fast_stage_L0.hef` | The compiled model. 480×640 grayscale in, two outputs: `resize1` (next pyramid stage's image, ignored here) and `activation1` (corner heatmap). |
| `CMakeLists.txt` | `find_package(HailoRT REQUIRED)`, links `HailoRT::libhailort`. |

The Hailo wrapper creates the `VDevice` with `group_id="SHARED"` +
`HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN` so the chip can be shared with other
processes and `HAILO_MONITOR=1` can publish stats.

---

## 6. Troubleshooting

**`fatal error: hailo/hailort.hpp: No such file or directory`**
HailoRT headers not installed. `sudo apt install hailort` (or follow Hailo's
install guide for your distro).

**`find_package(HailoRT)` fails**
`/usr/lib/cmake/HailoRT/HailoRTConfig.cmake` is missing — reinstall the
`hailort` deb package.

**`[HailoFeatureExtractor] VDevice::create failed`**
No accessible Hailo device. Check:
```bash
hailortcli scan
lsmod | grep hailo                # hailo_pci should be loaded
dmesg | grep -i hailo | tail
```

**Level-0 path tally shows `fallback_init` > 0**
The HEF couldn't be loaded. The error preceding the tally explains why
(missing file, wrong arch, etc.).

**Level-0 path tally shows `fallback_shape` > 0**
The HEF's output dimensions don't match `mvImagePyramid[0]`. This HEF expects
480×640 input. If you're running on a different camera resolution either
recompile the HEF for that resolution, or set the YAML's `Camera.newWidth` /
`Camera.newHeight` to 640 / 480.

**Pangolin/Qt segfault during shutdown**
Long-standing Pangolin bug independent of this fork. Disable the viewer
(`System(..., false)`) for headless runs; otherwise the trajectory is still
saved correctly before the segfault and exit-code 139 is benign.

**`[HailoRT] read_async() was provided an unaligned buffer`**
Performance hint — our output buffers aren't page-aligned. Doesn't affect
correctness. To silence, switch the output buffers to `posix_memalign(_, 16384, _)`
or `mmap`, matching the `page_aligned_alloc` pattern in
`hailo-apps/hailo_apps/cpp/common/hailo_infer.cpp`.

---

## 7. Useful commands

```bash
# Parse an HEF to see its I/O shapes
hailortcli parse-hef Hailo/dense_fast_stage_L0.hef

# Compare two HEFs
sha256sum Hailo/dense_fast_stage_L0.hef /path/to/new.hef

# Extract a HEF from a HAR archive
tar -xf model_compiled.har -C /tmp/extract/

# Watch which frames went through Hailo (last lines of stderr)
./Examples/RGB-D/rgbd_tum ... 2>&1 | grep "level-0 path tally" -A6
```
