"""Disentangle selection vs representation in the k x k x k game: at fixed k=4, does a
BIGGER buffer raise the reach cap? If yes -> representation ceiling. If flat -> selection."""
from scale_spike2 import run_n
print("k=4: does more buffer raise the cap? (H threshold ~3 since 4^3=64)\n")
for H in [1, 2, 3, 4]:
    run_n(4, H=H, seeds=6, episodes=2000, batch=64)
