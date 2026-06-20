# HPC Tensor Runtime

A high-performance tensor runtime built from scratch in C++/CUDA — a miniature
version of the kind of execution engine that sits behind frameworks like
PyTorch or TensorRT. The goal is depth, not breadth: every operator,
allocator, and scheduler in this project is hand-written, with a focus on
HPC fundamentals (memory layout, cache behavior, vectorization) and
systems design (execution graphs, scheduling, kernel dispatch).

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| 1 | Core Tensor Library (shape, strides, memory) | 🚧 In progress |
| 2 | Operator Kernels (MatMul, Reduce, Softmax, Conv — CPU + CUDA) | ⬜ Not started |
| 3 | Computation Graph (DAG, dependency tracking, scheduling) | ⬜ Not started |
| 4 | Runtime Scheduler (thread pool, CUDA streams, work stealing) | ⬜ Not started |
| 5 | Graph Optimizations (operator fusion, kernel auto-selection) | ⬜ Not started |

## Project structure

```
.
├── .devcontainer/      # VS Code Dev Container config
├── docker/             # Dockerfile for the dev/build environment
├── include/            # Public headers (Tensor class, etc.)
├── src/core/           # Core library implementation
├── tests/              # GoogleTest unit tests
├── CMakeLists.txt      # Build configuration
└── README.md
```

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (running)
- [VS Code Insiders](https://code.visualstudio.com/insiders/) with the
  **Dev Containers** extension installed

No local install of CMake, Ninja, or a C++ compiler is required — everything
needed to build the project lives inside the dev container.

## Building the project

The entire dev environment is containerized and isolated from your host
machine — no manual `git clone` on the host, no local toolchain setup.

### 1. Open the project in an isolated container

In VS Code Insiders:

```
Ctrl+Shift+P → "Dev Containers: Clone Repository in Container Volume…"
→ https://github.com/AkashRai242001/hpc-tensor-runtime
```

VS Code will build the image from `docker/Dockerfile`, create an isolated
volume, clone the repo into it, and open a window connected to the
container.

**Verify:** the bottom-left status bar shows `Dev Container: hpc-tensor-dev`.
Open a terminal (it's already a container shell) and run:

```bash
pwd        # /workspaces/hpc-tensor-runtime
ls         # CMakeLists.txt, include/, src/, tests/, docker/, ...
```

### 2. Verify the toolchain

```bash
cmake --version
ninja --version
g++ --version
git --version
```

Each should print a version with no "command not found" errors.

### 3. Create the build directory (parallel to the repo)

```bash
mkdir -p /workspaces/cmake_build
```

**Verify:**

```bash
ls /workspaces
# hpc-tensor-runtime  cmake_build
```

### 4. Configure with CMake + Ninja

```bash
cmake -S /workspaces/hpc-tensor-runtime -B /workspaces/cmake_build -G Ninja
```

**Verify:** the output ends with `Build files have been written to:
/workspaces/cmake_build`, and:

```bash
ls /workspaces/cmake_build
# build.ninja  CMakeCache.txt  CMakeFiles  ...
```

### 5. Build and run the tests

```bash
cd /workspaces/cmake_build
ninja
ctest --output-on-failure
```

**Verify:** `ninja` compiles with no errors; `ctest` reports `100% tests
passed`.

### 6. Install

```bash
ninja install
```

**Verify:** check the install prefix (default `/usr/local` unless overridden)
for the headers and library:

```bash
ls /usr/local/include | grep tensor
ls /usr/local/lib | grep tensor
```

### 7. Package

```bash
ninja package_all
```

**Verify:**

```bash
ls /workspaces/cmake_build/*.tar.gz
```

A packaged archive should now exist in the build directory.

## License

_Not yet specified._
