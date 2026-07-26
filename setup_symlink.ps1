# Run this script in PowerShell as Administrator (if required by your system for symlinks)
# This will link the VRI_Core library into your global Arduino libraries folder.

$ArduinoLibPath = "$env:USERPROFILE\Documents\Arduino\libraries\VRI_Core"
$SourcePath = Resolve-Path ".\firmware\libraries\VRI_Core"

if (Test-Path $ArduinoLibPath) {
    Write-Host "Symlink already exists at $ArduinoLibPath"
} else {
    New-Item -ItemType SymbolicLink -Path $ArduinoLibPath -Target $SourcePath
    Write-Host "Successfully created symlink for VRI_Core!"
}
