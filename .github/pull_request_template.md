## Scope checklist
- [ ] Only intended files and modules are changed
- [ ] Scope matches the task and avoids unrelated implementations

## Tests run
- [ ] `pytest tests/test_imports.py tests/test_cli_help.py -q`
- [ ] `ruff check src tests`

## Compliance
- [ ] No fake metrics or claimed benchmark reproduction
- [ ] No real data / checkpoint / large artifact committed
