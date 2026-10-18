# AVCS repository structure

## Repository boundaries

This repository contains several related Python packages. They remain
top-level siblings because they have different responsibilities and public
entry points:

| Package | Responsibility |
| --- | --- |
| `avcs/` | System 1 perception, action orchestration, desktop runtime, training, and SDK |
| `forge/` | Reusable vision/cortex primitives and the Forge framework |
| `avcs_sdk/` | SDK adapters for external repositories and integrations |
| `avcs_pilot/` | Pilot workflow and acceptance tooling |
| `avcs_m7_validation/` | Versioned validation and evidence tooling |
| `benchmarks/` | Reproducible benchmarks and evaluation harnesses |

`avcs` may depend on lower-level `forge` components. `forge` must not import
`avcs`; this keeps the visual primitives usable independently. Training,
benchmark, pilot, and validation packages are not part of the AVCS core runtime
layer.

## AVCS core packages

```text
avcs/
  perception/       Pixel models, focus, grounding, and visual events
  control/          System 1 orchestration, motor control, and verification
  runtime/          Capture, observation, streaming, and IPC
  memory/           Persistent visual scene state
  stream/           Reflex and neural telemetry streams
  drivers/          Optional System 2 provider adapters
  server/           HTTP API and System 2 text client
  mcp/              Model Context Protocol server
  auth/             Optional license-token utilities
  billing/          Usage-meter support
  dataset/          Dataset loaders and synthetic dataset generation
  training/         Recording, training, and evaluation tools
```

The perception, control, and runtime modules were grouped into these
subpackages. Existing module paths such as `avcs.human_eye` and
`avcs.controller` remain compatibility re-exports; new internal code should
import from the owning subpackage, for example
`avcs.perception.human_eye` or `avcs.control.controller`.

## Import and execution rules

- Importing `avcs` exposes the stable public API lazily. Import the specific
  subpackage/module for new code instead of depending on root-level exports.
- Package import must not mutate `sys.path` or eagerly import the model stack.
- Run modules from the repository root (or install the package) so Python can
  resolve the sibling packages. Do not rely on `avcs.__init__` to repair an
  external working directory.
- CLI entry points are declared in `pyproject.toml`; preserve their target
  modules or update the entry points and test them in the same change.
- `runs/`, checkpoints, captured demonstrations, build output, and benchmark
  results are data/artifacts, not Python source packages. Do not relocate or
  delete them as part of a source-layout change.

## Migration boundary

This is an incremental layout, not a complete repository rewrite. Existing
public import paths and console commands remain available while callers adopt
the new package paths. Root-level demos, training, benchmarks, and separate
product packages are intentionally outside the current AVCS-core move set.
