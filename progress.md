# Progress

## Done

- Implemented a toy example of the training loop in `flask_server/worker.py` using a mock dataset and a small GPT model.
- Integrated Megatron-Core for the training loop, including distributed training setup.
- Implemented the `forward`, `fwdbwd`, and `optim_step` functions based on the Megatron-Core examples.
- Implemented a placeholder for the calculation of `grad_norm`, `weight_norm`, and `update_norm`.

## Next Steps

- [ ] **Flesh out the training loop:**
    - [ ] Replace the mock dataset with a real dataset from the `tinker-cookbook`.
    - [ ] Replace the small GPT model with a real model from the `tinker-cookbook`.
    - [ ] Implement the `_call_megatron` function to use the loaded model for sampling, instead of calling out to a separate server.
- [ ] **Implement persistent state:**
    - [ ] Replace the in-memory `futures_store` with a persistent database (e.g., SQLite, PostgreSQL).
    - [ ] Implement a persistent store for LoRA adapters.
- [ ] **Implement LoRA:**
    - [ ] Implement the `add_lora` and `remove_lora` functions in `worker.py`.
    - [ ] Integrate LoRA loading and saving from the `tinker-cookbook`.
- [ ] **Implement weight management:**
    - [ ] Implement the `load_weights`, `save_weights`, and `save_weights_for_sampler` functions in `worker.py`.
    - [ ] Integrate checkpointing utilities from the `tinker-cookbook`.
- [ ] **Implement the `/get_info` endpoint:**
    - [ ] Implement the logic to retrieve model and adapter information.
- [ ] **Integrate `tinker-cookbook`:**
    - [ ] Integrate the `completers.py` module for different sampling strategies.
    - [ ] Integrate the `hyperparam_utils.py` module for managing hyperparameters.
    - [ ] Integrate the `tokenizer_utils.py` module for handling tokenization.
    - [ ] Integrate the various recipes in the `recipes` directory.
