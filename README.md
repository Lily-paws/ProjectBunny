# ProjectBunny (fork)

A fork of [StarBobis/ProjectBunny](https://github.com/StarBobis/ProjectBunny), a 3Dmigoto fork with DX11 and DX12 support.

The upstream repository is the canonical project. This fork focuses on refactoring the 3Dmigoto integration, particularly the DX12 path and its runtime performance. It is a working fork, not a separate release.

## Changes in this fork

TBA

## Potential changes

Work is targeted at the performance areas already identified in the DX12 path:

- Heap relevance gating — skip descriptor work when a heap is not actually relevant to the operation
- Reducing the self-overhead of the timing instrumentation
- Caching of the prepare stage (prepared state and masks)
- Trimming instrumentation out of the PSO/RSig lookup paths
- Optimising the command-submission and resource-transition paths
- Caching repeated CPU-side queries (e.g. shader modification-time checks)

None of this touches gameplay or features.

## Relationship to upstream

This fork is based on the upstream project and tracks it. Where performance-related work touches the DX12 layer the code may drift, though the goal is to keep it mergeable.

## Building

Windows, Visual Studio 2022 (Desktop development with C++ workload). CMake and Ninja are detected automatically if installed with VS.

```powershell
.\build_release_x64.ps1
```

Standard release build, producing the clean d3d12.dll.

```powershell
.\build_release_timing_x64.ps1
```

The same build with hot-path timing instrumentation enabled. This build is intended for log analysis rather than fps measurement, as the instrumentation itself adds overhead.

Both scripts copy the output DLLs to `x64\Release`. If `build_config.json` points at a game directory they deploy there as well; otherwise the deploy step is skipped with a warning.

## License

GPL-3.0, same as upstream. ProjectBunny by StarBobis; original 3Dmigoto by bo3b.
