# Getting Started — Hailo-accelerated ORB-SLAM3

This fork offloads **FAST keypoint detection and pyramid downscaling at levels
0–3** from CPU to a **Hailo-8** NPU. The four HEFs form a chain: each takes
its level's grayscale image and emits two outputs — a per-pixel corner-weight
heatmap (used to extract keypoints via a 3×3 NMS, replacing the cell-based
`cv::FAST` scan upstream uses), and a downscaled image fed as the next stage's
input (replacing `cv::resize(..., INTER_LINEAR)`).

The resized image from each stage is stored in `mvImagePyramid[level+1]`, so
the descriptor pipeline, keypoint orientation, and stereo SAD matching all
read the NPU's downscale rather than OpenCV's. Pyramid levels 4..N-1 still run
`cv::FAST` over a `cv::resize`-built image on CPU.

```
                +-------+     +-------+     +-------+     +-------+
   image  ----> | L0 hef| --> | L1 hef| --> | L2 hef| --> | L3 hef|
   (480x640)    +---+---+     +---+---+     +---+---+     +---+---+
                    | heat        | heat        | heat        | heat
                    v             v             v             v
                keypoints     keypoints     keypoints     keypoints
                (level 0)     (level 1)     (level 2)     (level 3)

   resize1 of stage N -> mvImagePyramid[N+1]  (no cv::resize for L1..L3)
```

The four HEFs are wired up at scaleFactor 1.2 (the default in `TUM1.yaml`):

| Level | Input dims | HEF | resize1 (→ next stage's input) |
|-------|-----------:|-----|-------------------------------|
| 0 | 480×640 | `Hailo/dense_fast_stage_L0.hef` | 400×533 |
| 1 | 400×533 | `Hailo/dense_fast_stage_L1.hef` | 333×444 |
| 2 | 333×444 | `Hailo/dense_fast_stage_L2.hef` | 278×370 |
| 3 | 278×370 | `Hailo/dense_fast_stage_L3.hef` | 231×309 (consumed by `cv::resize` for level 4 if N>4) |

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
New Map created with 832 points
median tracking time: 0.055...
mean tracking time:   0.057...

[ORBextractor] Hailo per-level path tally
  level | hailo_ok | fb_init | fb_run | fb_shape
  ------+----------+---------+--------+---------
    0   |      792 |       0 |      0 |        0
    1   |      792 |       0 |      0 |        0
    2   |      792 |       0 |      0 |        0
    3   |      792 |       0 |      0 |        0
```

`hailo_ok = total frames` on every row confirms every frame's keypoints at
levels 0–3 came from the NPU.

### GUI vs headless

`Examples/RGB-D/rgbd_tum.cc` controls the viewer via the 4th `System(...)` arg:

- `true`  → Pangolin window opens (needs a display; `DISPLAY=:0` on the Pi)
- `false` → headless; useful for benchmarking, no Qt segfault on shutdown

Rebuild with `cmake --build build --target rgbd_tum -j$(nproc)` after toggling.

---

## 4. Configuration

### `ORB_SLAM3_HAILO_L<N>_HEF`  (N = 0, 1, 2, 3)

Per-stage HEF path. Default is `Hailo/dense_fast_stage_L<N>.hef` relative to
cwd (all four are checked in on this branch). Override any of them:

```bash
export ORB_SLAM3_HAILO_L0_HEF=/abs/path/to/L0.hef
export ORB_SLAM3_HAILO_L1_HEF=/abs/path/to/L1.hef
# levels not overridden fall back to the defaults
./Examples/RGB-D/rgbd_tum ...
```

If a HEF fails to load (missing file, wrong arch, etc.), that level alone
falls back to `cv::FAST` on CPU — the other levels keep running on the NPU.

Constraints on each HEF:
- Single UINT8 input matching `mvImagePyramid[level]` dims
  (480×640, 400×533, 333×444, 278×370 for L0..L3 at scaleFactor 1.2)
- One UINT8 output with the same HxW as the input and 1 feature
  (the corner heatmap — recognised by matching dims; name is also `activation1`)
- Extra outputs are tolerated and ignored

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
| `src/ORBextractor.cc` — `ComputePyramid()` | For levels 1..3, copies the previous stage's `resize1` into `mvImagePyramid[level]` instead of running `cv::resize`. Triggers `Run()` for each stage so the heatmap is cached for `ComputeKeyPointsOctTree`. Per-frame outcomes stored in `mHailoOutcomeThisFrame`. |
| `src/ORBextractor.cc` — `ComputeKeyPointsOctTree()` | Reads each stage's cached heatmap (no second inference). For levels without a wrapper (4..N-1) or with a runtime failure, uses the upstream `cv::FAST` path. |
| `Hailo/dense_fast_stage_L<N>.hef` | One compiled model per pyramid level. Each consumes its level's grayscale image and emits `resize1` (consumed by stage N+1 / `mvImagePyramid[N+1]`) and `activation1` (corner heatmap at the level's resolution). |
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

**Path tally shows `fb_init > 0` for some level**
That level's HEF couldn't be loaded. The error preceding the tally explains
why (missing file, wrong arch, etc.). That level falls back to `cv::FAST` on
CPU; other levels keep running on the NPU.

**Path tally shows `fb_shape > 0` for some level**
The HEF's output dimensions don't match `mvImagePyramid[level]`. The bundled
HEFs target ORB-SLAM3's standard pyramid at scaleFactor 1.2 (480×640 →
400×533 → 333×444 → 278×370). If you're running on a different camera
resolution or scale factor, recompile the HEFs for those dims, or set the
YAML's `Camera.newWidth` / `Camera.newHeight` to 640 / 480.

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
