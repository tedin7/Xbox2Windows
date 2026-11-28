<#
.SYNOPSIS
Stages a small, disposable executable directory for an offline startup run.

.DESCRIPTION
Copies one executable and the DLLs beside it into a separate run directory. Package assets
remain in PackageDir and are supplied as run.ps1's -DataDir. No recursive copy is performed.
#>
param(
    [Parameter(Mandatory)] [string]$PackageDir,
    [Parameter(Mandatory)] [string]$Exe,
    [Parameter(Mandatory)] [string]$RunDir
)
$ErrorActionPreference = "Stop"

$package = (Resolve-Path -LiteralPath $PackageDir).Path
$source = if ([IO.Path]::IsPathRooted($Exe)) {
    (Resolve-Path -LiteralPath $Exe).Path
} else {
    (Resolve-Path -LiteralPath (Join-Path $package $Exe)).Path
}
$sourceDir = Split-Path -Parent $source
$provider = $null
$drive = $null
$run = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath(
    $RunDir, [ref]$provider, [ref]$drive)
if ($provider.Name -ne 'FileSystem') { throw "RunDir must be a filesystem path." }
if (-not (Test-Path -LiteralPath $source -PathType Leaf) -or [IO.Path]::GetExtension($source) -ine '.exe') {
    throw "Exe must name an executable file."
}

function Test-PathInside([string]$Path, [string]$Parent) {
    $prefix = $Parent.TrimEnd('\') + '\'
    return $Path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
}

# Resolve drive aliases and junctions before comparing package and staging paths.
# A destination may not exist yet, so resolve its nearest existing ancestor and
# append the remaining names without creating anything.
if (-not ("X2wStagingFinalPath" -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;

public static class X2wStagingFinalPath {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern SafeFileHandle CreateFileW(string name, uint access, uint share, IntPtr security,
        uint creation, uint flags, IntPtr template);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern uint GetFinalPathNameByHandleW(SafeFileHandle handle, StringBuilder path,
        uint length, uint flags);
    public static string Resolve(string name) {
        using (var handle = CreateFileW(name, 0, 7, IntPtr.Zero, 3, 0x02000000, IntPtr.Zero)) {
            if (handle.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error());
            var path = new StringBuilder(32768);
            uint length = GetFinalPathNameByHandleW(handle, path, (uint)path.Capacity, 0);
            if (length == 0) throw new Win32Exception(Marshal.GetLastWin32Error());
            if (length >= path.Capacity) throw new InvalidOperationException("Resolved path is too long.");
            return path.ToString();
        }
    }
}
'@
}
function Get-FinalStagingPath([string]$Path) {
    $missing = [System.Collections.Generic.List[string]]::new()
    $existing = $Path
    while (-not (Test-Path -LiteralPath $existing)) {
        $parent = Split-Path -Parent $existing
        if (-not $parent -or $parent -eq $existing) { throw "Cannot resolve staging path." }
        $missing.Insert(0, (Split-Path -Leaf $existing))
        $existing = $parent
    }
    $nativePath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($existing)
    $final = [X2wStagingFinalPath]::Resolve($nativePath)
    foreach ($part in $missing) { $final = [IO.Path]::Combine($final, $part) }
    return $final
}

if (-not (Test-PathInside $source $package)) {
    throw "Exe must be inside PackageDir."
}
if ($run.Equals($package, [StringComparison]::OrdinalIgnoreCase) -or
    (Test-PathInside $run $package) -or (Test-PathInside $package $run)) {
    throw "RunDir must be separate from PackageDir and its parent directories."
}

# Do not write through an existing junction/symlink or overwrite a previous package.
for ($ancestor = $run; $ancestor; $ancestor = Split-Path -Parent $ancestor) {
    if ((Test-Path -LiteralPath $ancestor) -and
        ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "RunDir must not pass through a reparse point."
    }
}
$packageFinal = Get-FinalStagingPath $package
$runFinal = Get-FinalStagingPath $run
$sourceFinal = Get-FinalStagingPath $source
if (-not (Test-PathInside $sourceFinal $packageFinal)) { throw "Exe must be inside PackageDir." }
if ($runFinal.Equals($packageFinal, [StringComparison]::OrdinalIgnoreCase) -or
    (Test-PathInside $runFinal $packageFinal) -or (Test-PathInside $packageFinal $runFinal)) {
    throw "RunDir must be separate from PackageDir and its parent directories."
}
if ((Test-Path -LiteralPath $run) -and
    (-not (Test-Path -LiteralPath $run -PathType Container) -or
     @(Get-ChildItem -LiteralPath $run -Force).Count)) {
    throw "RunDir must be absent or empty. Use a fresh directory for staging."
}

New-Item -ItemType Directory -Force -Path $run | Out-Null
Copy-Item -LiteralPath $source -Destination $run -Force
$dlls = @(Get-ChildItem -LiteralPath $sourceDir -File -Filter '*.dll' |
    Where-Object { $_.Extension -ieq '.dll' })
foreach ($dll in $dlls) {
    Copy-Item -LiteralPath $dll.FullName -Destination $run -Force
}
Write-Output "Staged executable and $($dlls.Count) adjacent DLLs in $run; package assets remain in $package."
