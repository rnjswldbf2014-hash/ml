# Builds rnjswldbf_2014/ml.d into <OutDir>/ml.pyd for testing.
# Usage: powershell -File tests/build.ps1 -OutDir <dir>
param(
    [Parameter(Mandatory=$true)][string]$OutDir
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$ldc  = Join-Path $root "ldc2\ldc2-1.42.0-windows-x64\bin\ldc2.exe"
$pylib = "$env:LOCALAPPDATA\Programs\Python\Python313\libs\python313.lib"
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$out = Join-Path $OutDir "ml.pyd"
$src = Join-Path $root "rnjswldbf_2014\ml.d"
$gpu = Join-Path $root "rnjswldbf_2014\gpu_cl.d"

& $ldc $src $gpu $pylib --O3 --release --shared --link-defaultlib-shared=false "-of=$out" 2>&1
if ($LASTEXITCODE -ne 0) { throw "ldc2 build failed (exit $LASTEXITCODE)" }

# clean up incidental link artifacts next to the .pyd
Remove-Item (Join-Path $OutDir "ml.obj"), (Join-Path $OutDir "ml.lib"), (Join-Path $OutDir "ml.exp"),
            (Join-Path $OutDir "gpu_cl.obj") -ErrorAction SilentlyContinue
Write-Output "built: $out"
