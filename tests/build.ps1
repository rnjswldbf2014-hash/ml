# Builds rnjswldbf_2014/ml.d into <OutDir>/ml.pyd for testing.
# Usage: powershell -File tests/build.ps1 -OutDir <dir>
#
# Retries, because this build fails intermittently (~5% of invocations) in a way
# that has nothing to do with the code. The failure looks like:
#
#   lld-link: error: could not open 'C:\Users\권�???Desktop\코딩 ?�로?�트??dml\
#             tests\_scratch\_module\ml.obj': no such file or directory
#
# Note the path in the message is MANGLED -- the Korean directory names came
# through as replacement characters. The object file is written fine; lld-link is
# handed a corrupted path for it. So this is a code-page handling glitch on the
# ldc2 -> lld-link hand-off with a non-ASCII project path, not a missing file and
# not anything about ml.d. There is no ASCII path available to build from (the
# user profile itself is non-ASCII), so the practical fix is to try again: the
# mangling is intermittent, and a retry has always succeeded.
#
# If this ever starts failing three times in a row, look at the path, not the code.
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

# NOTE on the two lines around the ldc2 call: with $ErrorActionPreference = "Stop",
# a native command writing ANYTHING to stderr becomes a terminating error in
# PowerShell 5.1, so the retry below never got to look at $LASTEXITCODE -- the
# first attempt threw straight out of the script. Dropping to "Continue" just for
# the call is what makes the exit code reachable. (Found the hard way: the retry
# was in place and silently never fired.)
$attempts = 3
$log = ""
for ($i = 1; $i -le $attempts; $i++) {
    $keep = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $log = & $ldc $src $gpu $pylib --O3 --release --shared --link-defaultlib-shared=false "-of=$out" 2>&1
        $code = $LASTEXITCODE
    } catch {
        $log = $_.ToString()
        $code = 1
    } finally {
        $ErrorActionPreference = $keep
    }
    if ($code -eq 0) { break }
    if ($i -eq $attempts) {
        Write-Output $log
        throw "ldc2 build failed $attempts times (last exit $code)"
    }
    Write-Output "build attempt $i failed, retrying"
    Start-Sleep -Milliseconds 400
}

# clean up incidental link artifacts next to the .pyd
Remove-Item (Join-Path $OutDir "ml.obj"), (Join-Path $OutDir "ml.lib"), (Join-Path $OutDir "ml.exp"),
            (Join-Path $OutDir "gpu_cl.obj") -ErrorAction SilentlyContinue
Write-Output "built: $out"
