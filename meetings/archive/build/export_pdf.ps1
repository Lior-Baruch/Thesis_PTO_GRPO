# Convert a .pptx to .pdf via PowerPoint COM (no LibreOffice needed; PowerPoint must be installed).
#
#   .\export_pdf.ps1 ..\2026-07-26\results_snapshot_2026-07-26.pptx
#   .\export_pdf.ps1 ..\2026-07-26\results_snapshot_2026-07-26.pptx -Png   # also dump slide PNGs
#
# Output .pdf lands next to the .pptx. -Png writes <name>_png\SlideN.PNG beside it (useful for
# eyeballing layout without opening PowerPoint).
param(
    [Parameter(Mandatory = $true)][string]$Pptx,
    [switch]$Png
)

$ErrorActionPreference = "Stop"
$in = (Resolve-Path $Pptx).Path
if (-not $in.EndsWith(".pptx")) { throw "expected a .pptx, got: $in" }
$out = [System.IO.Path]::ChangeExtension($in, ".pdf")

$app = New-Object -ComObject PowerPoint.Application
try {
    # Open(path, ReadOnly, Untitled, WithWindow)
    $pres = $app.Presentations.Open($in, $true, $false, $false)
    if ($Png) {
        $dir = [System.IO.Path]::ChangeExtension($in, $null).TrimEnd('.') + "_png"
        New-Item -ItemType Directory -Force $dir | Out-Null
        Get-ChildItem "$dir\*.PNG" -ErrorAction SilentlyContinue | Remove-Item -Force
        $pres.Export($dir, "PNG", 1600, 900)
        Write-Host "PNGs -> $dir"
    }
    $pres.SaveAs($out, 32)   # 32 = ppSaveAsPDF
    $pres.Close()
} finally {
    $app.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}

"PDF -> {0}  ({1:N0} KB)" -f $out, ((Get-Item $out).Length / 1KB)
