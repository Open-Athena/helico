# AGENTS.md

Rules and conventions for AI agents working in this repo. Claude Code, Codex,
Cursor, and similar tools should treat these as overriding defaults.

## Hard rules

### Never monkey-patch

Do not replace functions, methods, or attributes of imported modules at
runtime to work around a bug or limitation. Monkey-patches are silent,
non-local, and frequently don't work the way you expect — they only
affect attribute lookups on the patched module's namespace, not
references already imported into other modules. They also vanish from
code-search and stay invisible to anyone reading the call site.

If a third-party library has a hard-coded behavior that doesn't fit our
needs:

1. Pad / preprocess inputs so the library's code path works (preferred)
2. Wrap or subclass the library's exposed API
3. Open an issue / contribute a patch upstream
4. As a last resort: vendor a small fork of the offending file with a
   clear explanation

If you cannot do any of those without significant engineering, **ask
the user** before introducing a workaround that hides behavior.

History: a session in 2026-04 tried to monkey-patch
`torch.nn.attention.sdpa_kernel` to inject a MATH backend fallback for
cuequivariance's hardcoded SDPA priority list. The patch silently did
nothing because cuequivariance had already imported the function. The
correct fix was to pad token sequences in `collate_fn` so cuDNN's
flash-attn could compile a kernel.

### W&B runs always go to `timodonnell/helico`

All training runs must log to **`https://wandb.ai/timodonnell/helico`**
(`WANDB_PROJECT=helico`, entity follows from the
`helico-wandb-modal` Modal secret which is keyed to that account).

The default in `modal/train.py` is `HELICO_TRAIN_WANDB_PROJECT=helico`,
so launching with the standard env vars already routes correctly. Do
**not** set `WANDB_PROJECT`, `WANDB_ENTITY`, or
`HELICO_TRAIN_WANDB_PROJECT` to a different value when kicking off
a run — keeping all runs in one project is what makes the leaderboard
view (issue-tagged comparisons, x-axis sweeps) actually useful.

If you need to scope a one-off experiment that shouldn't pollute the
shared project, prefix the run name (`exp9-lrsweep-3e-4`,
`debug-cuda-oom`) — don't fork the project.
