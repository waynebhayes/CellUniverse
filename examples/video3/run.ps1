$dir=$(Get-Location)
Set-Location ../..
python "./src/main.py" `
    --frame_first 22 `
    --frame_last 200 `
    --input "$dir/input/%d.jpg" `
    --bestfit "$dir/output/bestfit" `
    --output "$dir/output" `
    --config "$dir/config.json" `
    --initial "$dir/initial.csv"  `
    --continue_from 22 `
    -ta 0 `
    -ts 1 `
    -te 0.000001 `
    --no_parallel --graySynthetic --global_optimization
if($LASTEXITCODE -ne 0){
    Write-Output "Python quit unexpectedly!"
}
Set-Location $dir