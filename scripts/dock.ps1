<#
dock.ps1 - run commands in the n2-sqd container (repo bind-mounted at /work).
Usage:
  scripts/dock.ps1 python scripts/verify_rotation.py submission
  scripts/dock.ps1 jupyter            # JupyterLab on http://localhost:8888
Thread counts default to 16 inside the image; override host-side, e.g.
  $env:OMP_NUM_THREADS = 4; scripts/dock.ps1 python scripts/optimise.py ...
Host-set OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS are forwarded.
#>
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Image = "n2-sqd:latest"
if ($Args.Count -eq 0) { Write-Error "usage: dock.ps1 <command...> | dock.ps1 jupyter"; exit 1 }
$envArgs = @()
foreach ($v in @("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")) {
  $val = [Environment]::GetEnvironmentVariable($v)
  if ($val) { $envArgs += @("-e", "$v=$val") }
}
if ($Args[0] -eq "jupyter") {
  docker run --rm @envArgs -v "${RepoRoot}:/work" -w /work -p 8888:8888 $Image `
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/work
} else {
  docker run --rm @envArgs -v "${RepoRoot}:/work" -w /work $Image @Args
}
