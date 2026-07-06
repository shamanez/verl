# vast-provision — operator maintenance notes

Not loaded by agents (SKILL.md is the agent contract). Read this when you
maintain the Vast template or debug a weird box.

## Template maintenance footguns

- `vastai update template` NULLS every field you omit — it can silently brick
  the template. Prefer **recreate**: build a new template with the full field
  set (image / docker_options / onstart from `templates.json` + `onstart.verl-vllm020.sh`),
  then update `templates.json` with the new hash. The hash is content-derived —
  an identical rebuild yields the same hash.
- `vastai delete template` takes the numeric `template_id`, NOT the `hash_id`.
- The CLI has no `show templates`; list your own with
  `vastai search templates 'creator_id in [<your-user-id>]' --raw`.
- A Vast template is per-account: the team account cannot use the private hash
  (create 400s). Record a team-owned copy as `team_hash_id` in `templates.json`.

## Debugging boxes

- `nvidia-smi` inside the container reports the forward-compat CUDA ceiling,
  not the host driver's native version. A too-loose `cuda_max_good` filter
  surfaces as CUDA `Error 803` at the FIRST kernel launch (imports succeed).
- Vast silently strips `--cap-add=SYS_ADMIN` and ignores `--shm-size` from
  template docker options — only the `-e` env vars reliably take effect.
- pids.max ≤ 2048 hosts SIGABRT under FSDP+vLLM (~1700+ threads needed);
  run.sh now auto-probes and destroys+advances on those hosts.
