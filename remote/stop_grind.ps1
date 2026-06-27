# Reliably stop the drone grind. MUST kill the run_all.sh supervisor bash loops
# BEFORE the python workers, else run_worker resurrects them (orphan stacking).
# Use this instead of "kill python" or TaskStop alone.
$sup = Get-CimInstance Win32_Process -Filter "Name='bash.exe'" |
       Where-Object { $_.CommandLine -match 'run_all|run_worker' }
$sup | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2
$py = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
      Where-Object { $_.CommandLine -match 'pool_worker|random_search' -or $_.ExecutablePath -match 'torch-gpu' }
$py | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2
$b = (Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -match 'run_all' }).Count
$p = (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.ExecutablePath -match 'torch-gpu' -or $_.CommandLine -match 'pool_worker' }).Count
"stopped supervisors=$($sup.Count) workers=$($py.Count); remaining grind-bash=$b torch-python=$p"