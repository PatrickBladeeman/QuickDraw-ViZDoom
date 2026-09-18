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
