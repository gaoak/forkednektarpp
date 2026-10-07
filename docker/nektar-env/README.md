# `nektar-env` image

This image is designed to provide a build environment for Nektar++ based on the
Debian 12 (bookworm) build image. It installs all libraries to enable Nektar++
to be compiled with most/all third-party dependencies turned on. In particular
we install the following development libraries:

- Boost
- TinyXML
- LAPACK/BLAS
- OpenMPI (enables `NEKTAR_USE_MPI`)
- FFTW (enables `NEKTAR_USE_FFT`)
- Python (enables `NEKTAR_BUILD_PYTHON`)
- HDF5 (enables `NEKTAR_USE_HDF5`)
- OCE, Triangle and TetGen (enables `NEKTAR_USE_MESHGEN`)
- PETSc (enables `NEKTAR_USE_PETSC`)
- ARPACK (enables `NEKTAR_USE_ARPACK`)

## Building

No particular context is required to build this image -- use the command below,
or similar.

```sh
docker build -t nektarpp/nektar-env -f Dockerfile .
```

## Other environment images

The other dockerfiles for different operating systems and package lists are used
to provide environment images for the CI system. See `.gitlab-ci.yml` for
details.

## Cluster environment images

The `*-cluster-submit` jobs run on aarch64 GPU nodes, which cannot run the
images the other `*-build-env` jobs push -- those are built by x86-64 runners.
There is no Linux aarch64 runner, so `ubuntu-noble-default-arm64-build-env` and
`ubuntu-noble-full-arm64-build-env` cross-build them under QEMU on the x86-64
runners and push them under `-arm64` tags.
`.gitlab-ci/cluster-test.sh` pulls `env-ubuntu-noble-full-arm64`.

Two things about those jobs differ from the native ones, both in
`.build-env-template-arm64`:

- They run on narrower conditions (`.execution-conditions-arm64`): only a
  change under `docker/nektar-env/` or to `.gitlab-ci.yml` rebuilds them. An
  emulated build is slow, and a merge request touching `library/` cannot change
  what goes into these images. The cluster jobs therefore depend on them with
  `optional: true`, and in every other pipeline start straight away against the
  image already in the registry.
- The full stage retags the pulled `-arm64` default image to the name the
  `FROM` line of `Dockerfile_ubuntu_full` expects, and builds without `--pull`.
  That line carries no architecture, so without this the aarch64 image would be
  layered on the x86-64 default image.

### Building one by hand

Useful for testing a change to these Dockerfiles without waiting on a pipeline.
Run from the root of a checkout. `REGISTRY` is the project's container registry
path, as shown under Deploy > Container Registry in GitLab.

On an x86-64 machine, register QEMU's aarch64 handler first. This is a one-off
per machine and does not survive a reboot on most distributions; some
distributions package the same thing as `qemu-user-static`. On an aarch64
machine, skip it and drop `--platform linux/arm64` below.

```sh
docker run --privileged --rm tonistiigi/binfmt --install arm64
export REGISTRY=<registry path>
docker login $REGISTRY
```

Build the default stage. Tag it with the name the full stage's `FROM` line
expects, so that line resolves to this image rather than the x86-64 one in the
registry. It does not need pushing unless you want CI to pick it up.

```sh
sed -e "s %%OS_VERSION%% noble g" \
    docker/nektar-env/Dockerfile_ubuntu_default > Dockerfile
docker build --platform linux/arm64 -t $REGISTRY:env-ubuntu-noble-default .
```

Build the full stage. Note the absence of `--pull`, which the native CI job
does pass: here it would replace the image just tagged with the x86-64 one of
the same name.

```sh
sed -e "s %%OS_DISTRO%% ubuntu g" -e "s %%OS_VERSION%% noble g" \
    -e "s %%REGISTRY%% $REGISTRY g" \
    docker/nektar-env/Dockerfile_ubuntu_full > Dockerfile
docker build --platform linux/arm64 \
    --build-arg "PIP_FLAGS=--break-system-packages --ignore-installed" \
    -t $REGISTRY:env-ubuntu-noble-full-arm64 .
```

`PIP_FLAGS` matches what `ubuntu-noble-full-arm64-build-env` sets in
`.gitlab-ci.yml`; keep the two in step. Before pushing anything the cluster will
use, check it really is aarch64:

```sh
docker image inspect $REGISTRY:env-ubuntu-noble-full-arm64 \
    --format '{{.Architecture}}'
docker push $REGISTRY:env-ubuntu-noble-full-arm64
```
