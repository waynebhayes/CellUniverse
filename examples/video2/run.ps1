$dir=$(Get-Location)
Set-Location ../..
python "./src/main.py" `
    --frame_first 0 `
    --frame_last 100 `
    --input "$dir/input/original/%d.png" `
    --bestfit "$dir/output/bestfit" `
    --output "$dir/output" `
    --config "$dir/config.json" `
    --initial "$dir/input/original/initial.csv"  `
    --no_parallel --graySynthetic --global_optimization

if($LASTEXITCODE -ne 0){
    Write-Output "Python quit unexpectedly!"
}
Set-Location $dir