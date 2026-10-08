param(
    [string]$Main2Zip = "C:\Users\Lars\Downloads\main2_local_loss_292452_0.err.zip",
    [string]$OutputDir = ".\round_curve_artifacts",
    [int]$SmoothWindow = 5
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Moving-Average([double[]]$Values, [int]$Window) {
    $result = New-Object double[] $Values.Count
    $sum = 0.0
    for ($i = 0; $i -lt $Values.Count; $i++) {
        $sum += $Values[$i]
        if ($i -ge $Window) { $sum -= $Values[$i - $Window] }
        $count = [Math]::Min($i + 1, $Window)
        $result[$i] = $sum / $count
    }
    return $result
}

function Sample-Stats($Series, [int]$Index) {
    $values = @($Series | ForEach-Object { [double]$_[$Index] })
    $mean = ($values | Measure-Object -Average).Average
    $sumSquares = ($values | ForEach-Object { ($_ - $mean) * ($_ - $mean) } |
        Measure-Object -Sum).Sum
    [pscustomobject]@{ Mean = $mean; SD = [Math]::Sqrt($sumSquares / ($values.Count - 1)) }
}

$seeds = @(42, 43, 44)
$main = @{}
foreach ($seed in $seeds) {
    $path = Get-ChildItem -LiteralPath . -Filter "cifar10_main_seeds_288836_*.log" |
        Where-Object { Select-String -LiteralPath $_.FullName -Pattern "seed=$seed; rounds=200" -Quiet } |
        Select-Object -First 1
    if (-not $path) { throw "Missing Main log for seed $seed" }
    $bacc = New-Object System.Collections.Generic.List[double]
    $auc = New-Object System.Collections.Generic.List[double]
    foreach ($line in Get-Content -LiteralPath $path.FullName) {
        if ($line -match '^Round\s+(\d+)\s+\|\s+AUC:([0-9.]+).*BAcc:([0-9.]+)') {
            $auc.Add([double]$Matches[2]); $bacc.Add([double]$Matches[3])
        }
    }
    if ($bacc.Count -ne 200) { throw "Main seed $seed has $($bacc.Count) rounds" }
    $main[$seed] = @{ BAcc = Moving-Average $bacc.ToArray() $SmoothWindow;
                      AUC = Moving-Average $auc.ToArray() $SmoothWindow }
}

$main2 = @{}
$zip = [System.IO.Compression.ZipFile]::OpenRead($Main2Zip)
try {
    foreach ($seed in $seeds) {
        $name = "main2_local_loss_outputs/292452/seed${seed}/main2_local_loss_cifar10_seed${seed}.json"
        $entry = $zip.GetEntry($name)
        if (-not $entry) { throw "Missing Main2 JSON: $name" }
        $reader = New-Object System.IO.StreamReader($entry.Open())
        try { $payload = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
        $bacc = [double[]]$payload.history.bacc
        $auc = [double[]]$payload.history.auc
        if ($bacc.Count -ne 200) { throw "Main2 seed $seed has $($bacc.Count) rounds" }
        $main2[$seed] = @{ BAcc = Moving-Average $bacc $SmoothWindow;
                           AUC = Moving-Average $auc $SmoothWindow }
    }
} finally { $zip.Dispose() }

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$csvPath = Join-Path $OutputDir "main_vs_main2_round_curves.csv"
$rows = for ($round = 0; $round -lt 200; $round++) {
    $row = [ordered]@{ round = $round + 1 }
    foreach ($seed in $seeds) {
        $row["main_bacc_seed$seed"] = $main[$seed].BAcc[$round]
        $row["main_auc_seed$seed"] = $main[$seed].AUC[$round]
        $row["main2_bacc_seed$seed"] = $main2[$seed].BAcc[$round]
        $row["main2_auc_seed$seed"] = $main2[$seed].AUC[$round]
    }
    [pscustomobject]$row
}
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $csvPath

$width = 1400; $height = 590
$panels = @(
    @{ X=80; Y=80; W=610; H=420; Metric="BAcc"; Title="(a) Balanced Accuracy"; Min=.45; Max=.92 },
    @{ X=760; Y=80; W=610; H=420; Metric="AUC"; Title="(b) AUC"; Min=.50; Max=.97 }
)
$styles = @{
    Main  = @{ Color="#D55E00"; Fill="#D55E00" }
    Main2 = @{ Color="#0072B2"; Fill="#0072B2" }
}
function XCoord($panel, [int]$index) { $panel.X + $panel.W * $index / 199.0 }
function YCoord($panel, [double]$value) {
    $clipped = [Math]::Max($panel.Min, [Math]::Min($panel.Max, $value))
    $panel.Y + $panel.H * (1.0 - ($clipped - $panel.Min) / ($panel.Max - $panel.Min))
}
function Path-For($panel, [double[]]$values) {
    $parts = for ($i=0; $i -lt $values.Count; $i++) {
        $prefix = if ($i -eq 0) { "M" } else { "L" }
        "$prefix$([Math]::Round((XCoord $panel $i),2)),$([Math]::Round((YCoord $panel $values[$i]),2))"
    }
    $parts -join " "
}
function Band-For($panel, [double[]]$mean, [double[]]$sd) {
    $upper = for($i=0;$i -lt 200;$i++){ "L$([Math]::Round((XCoord $panel $i),2)),$([Math]::Round((YCoord $panel ($mean[$i]+$sd[$i])),2))" }
    $lower = for($i=199;$i -ge 0;$i--){ "L$([Math]::Round((XCoord $panel $i),2)),$([Math]::Round((YCoord $panel ($mean[$i]-$sd[$i])),2))" }
    (($upper + $lower) -join " ") -replace '^L','M'
}

$svg = New-Object System.Text.StringBuilder
[void]$svg.AppendLine("<svg xmlns='http://www.w3.org/2000/svg' width='$width' height='$height' viewBox='0 0 $width $height'>")
[void]$svg.AppendLine("<rect width='100%' height='100%' fill='white'/>")
[void]$svg.AppendLine("<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.title{font-size:22px;font-weight:600}.axis{font-size:17px}.tick{font-size:14px;fill:#555}.legend{font-size:16px}</style>")
foreach($panel in $panels){
    [void]$svg.AppendLine("<text class='title' x='$($panel.X)' y='42'>$($panel.Title)</text>")
    for($g=0;$g -le 4;$g++){$v=$panel.Min+($panel.Max-$panel.Min)*$g/4;$y=YCoord $panel $v;[void]$svg.AppendLine("<line x1='$($panel.X)' y1='$y' x2='$($panel.X+$panel.W)' y2='$y' stroke='#E5E7EB' stroke-width='1'/><text class='tick' x='$($panel.X-12)' y='$($y+5)' text-anchor='end'>$('{0:F2}' -f $v)</text>")}
    foreach($tick in 1,50,100,150,200){$x=XCoord $panel ($tick-1);[void]$svg.AppendLine("<text class='tick' x='$x' y='$($panel.Y+$panel.H+25)' text-anchor='middle'>$tick</text>")}
    [void]$svg.AppendLine("<line x1='$($panel.X)' y1='$($panel.Y)' x2='$($panel.X)' y2='$($panel.Y+$panel.H)' stroke='#333'/><line x1='$($panel.X)' y1='$($panel.Y+$panel.H)' x2='$($panel.X+$panel.W)' y2='$($panel.Y+$panel.H)' stroke='#333'/>")
    foreach($version in 'Main','Main2'){
        $source=if($version -eq 'Main'){$main}else{$main2};$series=@($seeds|ForEach-Object{[double[]]$source[$_][$panel.Metric]});$mean=New-Object double[] 200;$sd=New-Object double[] 200;for($i=0;$i -lt 200;$i++){$s=Sample-Stats $series $i;$mean[$i]=$s.Mean;$sd[$i]=$s.SD}
        $band=Band-For $panel $mean $sd;[void]$svg.AppendLine("<path d='$band Z' fill='$($styles[$version].Fill)' opacity='0.12'/>")
        foreach($seed in $seeds){$path=Path-For $panel ([double[]]$source[$seed][$panel.Metric]);[void]$svg.AppendLine("<path d='$path' fill='none' stroke='$($styles[$version].Color)' stroke-width='1' opacity='0.22'/>")}
        $meanPath=Path-For $panel $mean;[void]$svg.AppendLine("<path d='$meanPath' fill='none' stroke='$($styles[$version].Color)' stroke-width='4'/>")
    }
    [void]$svg.AppendLine("<text class='axis' x='$($panel.X+$panel.W/2)' y='$($panel.Y+$panel.H+62)' text-anchor='middle'>Communication round</text>")
    [void]$svg.AppendLine("<text class='axis' transform='translate($($panel.X-58),$($panel.Y+$panel.H/2)) rotate(-90)' text-anchor='middle'>$($panel.Metric)</text>")
}
[void]$svg.AppendLine("<line x1='510' y1='555' x2='565' y2='555' stroke='#D55E00' stroke-width='4'/><text class='legend' x='575' y='561'>Main mean ± sample SD</text>")
[void]$svg.AppendLine("<line x1='805' y1='555' x2='860' y2='555' stroke='#0072B2' stroke-width='4'/><text class='legend' x='870' y='561'>Main2 mean ± sample SD</text>")
[void]$svg.AppendLine("</svg>")
$svgPath = Join-Path $OutputDir "main_vs_main2_round_curves.svg"
[System.IO.File]::WriteAllText((Join-Path (Resolve-Path $OutputDir) "main_vs_main2_round_curves.svg"), $svg.ToString(), [System.Text.UTF8Encoding]::new($false))
Write-Host "svg=$svgPath"
Write-Host "csv=$csvPath"
Write-Host "seeds=42,43,44; rounds=200; smoothing_window=$SmoothWindow"
