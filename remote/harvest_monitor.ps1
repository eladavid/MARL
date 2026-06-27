# Sample the resilient grind's harvest rate: finished H1 candidates + supervisor retries
# (crashes) per minute. Writes HARVEST.log. Exits after $Minutes samples.
param(
  [string]$Py  = "C:\Users\eladdavid1\AppData\Local\anaconda3\envs\torch-gpu\python.exe",
  [string]$Out = "C:\Users\eladdavid1\mac_reinforce\results_remote",
  [int]$Minutes = 12
)
$log = Join-Path $Out "HARVEST.log"
Set-Content -Path $log -Value "minute  H1_done  retries(crashes)  rate/min" -Encoding ascii
$counter = @"
import glob,pickle
t=0
for p in glob.glob(r'$Out\part_H1_*.pkl'):
    try: t+=len(pickle.load(open(p,'rb'))['ratios'])
    except: pass
print(t)
"@
$prev = 0
for ($m = 1; $m -le $Minutes; $m++) {
  Start-Sleep -Seconds 60
  $done = [int](& $Py -c $counter 2>$null)
  $retries = 0
  Get-ChildItem (Join-Path $Out "log_*.log") -ErrorAction SilentlyContinue | ForEach-Object {
    $retries += (Select-String -Path $_.FullName -Pattern 'supervisor.*retry' -ErrorAction SilentlyContinue).Count
  }
  $rate = $done - $prev; $prev = $done
  $line = "{0,5}  {1,7}  {2,15}  {3,8}" -f $m, $done, $retries, $rate
  Add-Content $log $line
}
Add-Content $log "DONE"
