# QuickDraw QA

Run the checkout-local static gate from the repository root:

```powershell
python -m pip install -r requirements-quickdraw.txt
python -m quickdraw_vizdoom.qa run --contract contracts/basic-v1.json --profile dev
```

The `dev` profile validates contracts without launching ViZDoom or making external
calls. The integration gate launches one process and validates the saved canonical
trace:

```powershell
python -m quickdraw_vizdoom.qa run --contract contracts/basic-v1.json --profile integration
```

Watch a fresh micro-learning agent train in the actual Doom window:

```powershell
.venv\Scripts\python.exe -m quickdraw_vizdoom.watch --delay 0.4
```

Watch the visual-generalization profile instead:

```powershell
.venv\Scripts\python.exe -m quickdraw_vizdoom.watch_generalization --delay 0.4
```

The agent controls the game; the terminal shows its action, reward, selection
source (`random` or `greedy`), and recent accuracy. A random
action can coincide with the greedy action and is still labeled exploration.
This tiny task alternates target-present and target-absent episodes,
each lasting one decision, with movement disabled. The demonstration runs 512
training decisions followed by 100 greedy held-out episodes. `--delay` sets the
viewing pace; Ctrl+C in the terminal stops the run. QA profiles remain headless.
The generalization watcher also prints the current visual split and variant.
