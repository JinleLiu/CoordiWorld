## Scope checklist
- [ ] Only intended files and modules are changed
- [ ] Scope matches the task and avoids unrelated implementations
- [ ] README/docs updated when behavior or usage changed

## Tests run
- [ ] `pytest tests/test_imports.py tests/test_cli_help.py tests/test_env_config.py -q`
- [ ] `pytest tests/test_data_registry.py tests/test_synthetic_dataset.py tests/test_jsonl_adapter.py tests/test_validate_data_cli.py tests/test_candidate_pool.py -q`
- [ ] `ruff check src tests`

## Compliance
- [ ] No fake metrics or claimed benchmark reproduction
- [ ] No real data / checkpoint / large artifact committed
- [ ] No private server paths, API keys, or `.env` files committed
