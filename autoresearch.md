# autoresearch

This file is written to be handed directly to Codex.

You are running autonomous research inside the current parameter-golf repository.

The old `prepare.py` / `train.py` workflow does not apply here. The current local research loop is built directly around `train_gpt.py`, with a compressed 16MB submission artifact and quantized round-trip evaluation.

## Codex Operating Contract

Assume the current working directory is the repository root.

Operate with these rules:

- Use exact shell commands and keep them non-interactive.
- Prefer small, controlled edits to `train_gpt.py` over broad rewrites.
- Redirect long command output to log files instead of streaming it into the chat.
- Do not stop to ask for confirmation once setup is complete.
- Do not commit `results.tsv`, `logs/`, or scratch outputs.
- When a run is worse than the best kept commit, revert the code state back to the best kept commit and continue.
- Report concise progress, but spend most effort on running experiments rather than narrating them.

## Mission

Primary objective:

- Minimize the final quantized validation score printed as `final_int8_zlib_roundtrip_exact ... val_bpb:...`.

Hard constraints:

- The compressed submission artifact must stay at or below `16_000_000` bytes.
- Do not modify the dataset, tokenizer assets, or evaluation semantics.
- Do not add new dependencies.

Secondary objectives:

- Keep peak VRAM reasonable.
- Prefer simpler changes when score is tied or nearly tied.
- Favor ideas that improve post-quant behavior, not just pre-quant training loss.

Current architectural bias:

- Treat the existing Plan C direction as the default mainline: unique front boundary, shared recurrent core, unique back boundary, lightweight loop alignment, QAT-aware training.
- You may deviate if the evidence is strong, but do not randomly thrash between unrelated ideas.

## Setup

When starting a fresh autonomous run, do this exactly:

1. **Choose a run tag**: use a tag based on today's date, for example `mar30` or `mar30-gpu0`. The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: create `autoresearch/<tag>` from the intended base branch, normally `main` unless the human said otherwise.
3. **Read the in-scope files**: read these files before making decisions:
   - `README.md` — challenge rules, current leaderboard context, dataset download instructions.
   - `train_gpt.py` — the main file you tune.
   - `docs/plan_c.md` — the current architecture direction and experimental framing.
   - `docs/plan.md` — broader recurrent / quantization design context.
   - `pyproject.toml` — dependency constraints.
4. **Verify data exists**: confirm these paths exist and are populated:
   - `./data/datasets/fineweb10B_sp1024/`
   - `./data/tokenizers/fineweb_1024_bpe.model`
   If they are missing, tell the human to populate them with `python3 data/cached_challenge_fineweb.py --variant sp1024`.
5. **Initialize results.tsv**: create `results.tsv` with only the header row shown below. Do not commit this file.
6. **Start immediately**: once setup is valid, run the baseline without waiting for more approval.

Suggested setup commands:

```bash
git checkout main
git pull --ff-only
git checkout -b autoresearch/<tag>
test -d ./data/datasets/fineweb10B_sp1024
test -f ./data/tokenizers/fineweb_1024_bpe.model
printf 'commit\tval_bpb\tval_bpb_1000\tmemory_gb\tartifact_mb\tstatus\tdescription\n' > results.tsv
```

## What You Can Change

- Modify `train_gpt.py`.
- Override environment variables at launch time to test alternate settings.

## What You Must Not Change

- Do not edit dataset files, tokenizer files, or cached artifacts under `data/`.
- Do not change the quantized artifact accounting logic or the final evaluation semantics.
- Do not add packages beyond what already exists in `pyproject.toml`.
- Do not use `run_planc_local_1gpu.sh` in this workflow. The default single-GPU settings now live in `train_gpt.py`.
- Do not commit `results.tsv`, `logs/`, or ad hoc scratch files.

## Baseline Run

The first run must establish a baseline using the current code and the defaults embedded in `train_gpt.py`.

Default local launch command:

```bash
mkdir -p logs
RUN_ID=<run_id> torchrun --standalone --nproc_per_node=1 train_gpt.py > logs/<run_id>.driver.log 2>&1
```

Notes:

- Use time-first run ids so logs sort chronologically, for example `20260331_014700_stageanchors`.
- `train_gpt.py` already writes its own structured log to `logs/<RUN_ID>.txt`.
- Redirecting stdout and stderr to `logs/<run_id>.driver.log` avoids flooding the agent context.
- The current built-in single-GPU defaults are: `MAX_WALLCLOCK_SECONDS=1800`, `TRAIN_BATCH_TOKENS=1048576`, `GRAD_ACCUM_STEPS=2`, `NUM_FRONT_BLOCKS=1`, `NUM_CORE_BLOCKS=3`, `NUM_CORE_LOOPS=1`, `NUM_BACK_BLOCKS=1`, `QAT_ENABLED=1`, `QAT_MODE=shared_early`, `VE_LAST_N=2`, `XSA_LAST_N=2`, `COLLECT_LOOP_STATS=1`.
- `NUM_CORE_LOOPS` now means the number of full sweeps over the shared core block bank. Effective core depth is `NUM_CORE_BLOCKS * NUM_CORE_LOOPS`.

Baseline command example:

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)_baseline
torchrun --standalone --nproc_per_node=1 train_gpt.py > logs/${RUN_ID}.driver.log 2>&1
```

## Metrics To Track

Primary score:

- `final_int8_zlib_roundtrip_exact val_loss:... val_bpb:...`

Important supporting metrics:

- `peak memory allocated: ... reserved: ...`
- `artifact_check:start ... total_bytes:... limit:16000000 ...`
- `Total submission size int8+zlib: ... bytes`
- `step:1000/20000 val_loss:... val_bpb:...`
- `architecture:planc ...`
- `align:enabled=...`
- `local_batch_tokens:...`

Optional metric:

- `final_ttt_sliding ... val_bpb:...` only matters when TTT is enabled. Treat it as supplementary unless the human explicitly says otherwise.

Useful extraction commands:

```bash
grep -E "step:1000/20000 val_loss|final_int8_zlib_roundtrip_exact|final_ttt_sliding|peak memory allocated:|artifact_check:start|Total submission size int8\+zlib:" logs/<run_id>.txt
```

If that returns no final score line, the run failed. Inspect the tail:

```bash
tail -n 80 logs/<run_id>.txt
```

Useful git extraction commands:

```bash
git rev-parse --short HEAD
git status --short
```

## Logging Results

When an experiment completes, append a row to `results.tsv`. Use tab separation, not commas.

Header:

```tsv
commit	val_bpb	val_bpb_1000	memory_gb	artifact_mb	status	description
```

Columns:

1. Short git commit hash, 7 chars.
2. Primary `final_int8_zlib_roundtrip_exact` `val_bpb`, or `0.000000` for crashes.
3. `step:1000/20000 ... val_bpb`, or `0.0000` if the run crashed or never reached step 1000.
4. Peak allocated memory in GB, rounded to one decimal place, or `0.0` for crashes.
5. Final `int8+zlib` submission size in decimal MB, rounded to three decimals, or `0.000` for crashes.
6. Status: `keep`, `discard`, or `crash`.
7. Short description of the hypothesis tested.

Example:

```tsv
commit	val_bpb	val_bpb_1000	memory_gb	artifact_mb	status	description
a1b2c3d	1.132450	1.2841	42.8	15.612	keep	baseline planc local 1gpu
b2c3d4e	1.129980	1.2817	43.1	15.744	keep	shift QAT earlier for shared core
c3d4e5f	1.131700	1.2839	45.6	15.731	discard	enable affine aligner on all loops
d4e5f6g	0.000000	0.0000	0.0	0.000	crash	increase core loops to 6 caused OOM
```

## High-Value Experiment Axes

Prefer ideas that fit the current codebase rather than restarting from scratch.

- Boundary/core/back depth allocation.
- `ALIGN_MODE`, `ALIGN_SCALE_CLAMP`, `ALIGN_MIX_INIT`.
- `CARRY_ENABLED` and `CARRY_INIT`.
- `QAT_MODE` and QAT start fractions.
- `VE_LAST_N` and `XSA_LAST_N` placement.
- Per-loop controls versus simplification of those controls.
- Batch size, grad accumulation, warmup, warmdown, LR, momentum, weight decay, EMA, SWA.
- Quantization-aware simplifications that reduce artifact size without hurting score.

## The Experiment Loop

The experiment runs on a dedicated branch such as `autoresearch/mar30`.

LOOP FOREVER:

1. Check the current git state and identify the best kept commit so far.
2. Pick one concrete hypothesis.
3. Make the smallest code or launch-config change needed to test that hypothesis.
4. Commit the experiment change. Do not include `results.tsv` or log files in the commit.
5. Launch the run with a unique `RUN_ID`.
6. Extract the primary score, memory, and artifact size from the log.
7. If the run crashed, inspect the stack trace and decide whether it is a trivial fix or a bad idea.
8. Append the result to `results.tsv`.
9. Keep the commit only if it improves the primary score while staying within the 16MB artifact limit and without an unacceptable memory regression.
10. If the score is worse, tied without compensating simplification, or over the artifact limit, revert to the previous kept commit and continue.

Recommended command skeleton:

```bash
git rev-parse --short HEAD
git add train_gpt.py
git commit -m "exp: <short hypothesis>"
mkdir -p logs
RUN_ID=$(date +%Y%m%d_%H%M%S)_<hypothesis>
torchrun --standalone --nproc_per_node=1 train_gpt.py > logs/${RUN_ID}.driver.log 2>&1
grep -E "step:1000/20000 val_loss|final_int8_zlib_roundtrip_exact|final_ttt_sliding|peak memory allocated:|artifact_check:start|Total submission size int8\+zlib:" logs/${RUN_ID}.txt
```

If the run is a loser, reset the code state back to the best kept commit on the autoresearch branch before starting the next idea.

Interpretation rules:

- The primary comparison metric is quantized round-trip `val_bpb`, not pre-quant loss.
- A better score that breaks the 16MB limit is a discard unless the human explicitly says to explore non-record settings.
- If two runs are effectively tied, prefer the simpler and cheaper one.
- If a change improves training behavior but worsens final quantized score, discard it by default.

## Crash Policy

- If the failure is a trivial bug introduced by the experiment, fix it and rerun.
- If the idea is fundamentally broken, log it as `crash`, revert, and move on.
- If a run exceeds the intended budget by a large margin or appears hung, kill it, log failure, and continue.

## Autonomy Rule

Once the loop begins, do not pause to ask whether you should continue. Keep iterating until the human explicitly interrupts you.

If you run out of ideas, re-read the current code and docs, inspect prior kept and discarded runs, and keep searching for better trade-offs. The job is autonomous experimental progress, not status reporting.

## Output Style For Codex

When you do write updates, keep them brief and operational:

- what hypothesis you are testing
- what command you are running
- whether the result was keep, discard, or crash

Do not write long essays between runs.
