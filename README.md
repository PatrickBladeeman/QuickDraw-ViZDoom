# QuickDraw-ViZDoom

QuickDraw-ViZDoom is a research fork of [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)
for testing goal-conditioned reinforcement learning and LLM advisers. The current
Basic task compares no goal conditioning, random goals, rule-based goals, and
LLM-selected goals on held-out target-hit episodes.

QuickDraw adds task wrappers, learning and QA code, and experiment reports to the
upstream ViZDoom platform. The [original ViZDoom README](docs/upstream/ViZDoom-README.md)
is preserved for upstream installation instructions, features, and licensing
information. See the [ViZDoom documentation](https://vizdoom.farama.org/) for its API.

All commands below run from the repository root. Learning experiments require
ViZDoom, NumPy, and PyTorch in the Python environment used to run them.

## Setup and QA

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

## Visual demonstrations

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

## Collection-teacher comparison

Compare collection teachers for the genuine two-goal Basic learner:

```powershell
.venv\Scripts\python.exe -m quickdraw_vizdoom.goal_teachers --teachers rule random --output artifacts/quickdraw/goal-teachers-v1.json
```

Each teacher trains the same three seeds for 512 decisions. Rule goals request
`ALIGN_WITHOUT_FIRE` until aligned, then `HIT_TARGET`; random goals use a separate
seeded stream. Goals persist until success or environment end. Every transition
still trains both goals. Frozen evaluation requests each goal directly on the
same 100 held-out reset seeds, and reports correct/swapped-goal success, returns,
paired action selections, replay coverage, and parameter hashes. This tests the
teacher's effect on collected training experience, not a director during evaluation.
All teachers receive the same privileged slot/ammunition/decision summary.
This is an exploratory toy task, with no claim of general LLM teaching.

For an LLM comparison, add `llm` to `--teachers` and supply `--llm-url` (the full
chat-completions URL) and `--llm-model`. A local loopback server or
`https://openrouter.ai/api/v1/chat/completions` is supported. OpenRouter reads
`OPENROUTER_API_KEY` from the environment; keys are never written to the report.
The CLI defaults to OpenRouter model `qwen/qwen3.5-9b` when `llm` is selected.
Requests use temperature 0 and at most 64 output tokens. Raw replies, latency,
provider usage when returned, and failed conditions are retained; invalid goals
or timeouts fail the condition without a fallback. Use `--evaluation-episodes 10`
for a small pilot. Output paths must be new so earlier results are preserved.

## Frozen-policy adviser comparison

To test adviser effectiveness during execution, use the four-arm runner:

```powershell
.venv\Scripts\python.exe -m quickdraw_vizdoom.adviser_evaluation --arms no_goal random rule --evaluation-episodes 10 --output artifacts/quickdraw/adviser-baselines-new.json
```

It trains two policies for each of three seeds: a no-goal policy with environment
reward, and a goal-conditioned policy with balanced alternating goals and the
existing exhaustive relabeling. Both use the same image encoder, nonlinear
six-action head, Adam optimizer, replay capacity, Double-DQN update, target sync
interval and 512-decision budget. The no-goal policy's two goal features are zero.
The no-goal arm runs without an adviser; random, rule and LLM use the same frozen
goal-conditioned policy per seed. Each adviser chooses a goal at reset or goal
success, and that goal persists until success or episode end.

The report records held-out target-hit rate, decisions to hit, environment return,
goal traces, adviser errors/latency/provider-reported cost, paired differences per
training seed, and unchanged parameter hashes. Failed episodes, including invalid
LLM responses/timeouts, count against hit rate and receive a decision cost of 300;
there is no rule fallback. The ten reset seeds give 30 rollouts per arm, with
three independently trained policies. This is a pilot, not a significance claim.

After setting `OPENROUTER_API_KEY`, reuse the saved policies to evaluate all four
arms, including the actual LLM, without retraining:

```powershell
.venv\Scripts\python.exe -m quickdraw_vizdoom.adviser_evaluation --checkpoint artifacts/quickdraw/adviser-baseline-pilot-20261001-v1-policies.pt --evaluation-episodes 10 --output artifacts/quickdraw/adviser-four-arm-pilot-20261001-v3.json
```

The LLM adapter defaults to `qwen/qwen3.5-9b`; endpoint/model/timeout can be set
with the same flags as the collection-teacher runner. Increase
`--evaluation-episodes` to 100 for the next evaluation; increase
`--training-decisions` when creating new policies if low-level competence is weak.
Random/rule/LLM comparisons isolate adviser choice. The no-goal comparison measures
the whole system: goal rewards, relabeling and symbolic adviser information also
differ. This experiment does not test whether LLM-guided training is better.

### Recorded methodology

The completed pilots used these settings; CLI defaults use 100 evaluation reset
seeds unless `--evaluation-episodes 10` is supplied:

| Setting | Pilot value |
| --- | --- |
| Independent training seeds | 32001, 34001, 35001 |
| Shared evaluation reset seeds | 36000-36009; disjoint from training resets |
| Training budget | 512 environment decisions per policy and seed |
| Replay warmup / sampled batch | 32 / 32 physical transitions |
| Exploration / discount | Training epsilon 0.2 / gamma 0.99; evaluation epsilon 0 |
| Optimizer / replay capacity | Adam, learning rate 0.001 / 10,000 transitions |
| Learning updates / target copies | 481 updates; hard copy every 100 updates (4 copies) |
| Episode limit | 300 decisions |
| Runtime | CPU, one PyTorch thread |

The low-level policy receives four stacked 84 x 84 grayscale frames, legal-action
masks, and a two-element one-hot goal (two zeros for no-goal). Its nonlinear head
predicts values for six movement/combat combinations: Stay/Left/Right crossed
with Idle/Shoot. Advisers receive only the privileged `position_slot`,
`target_slot`, `decision`, and `remaining_ammunition` summary; the rule selects
ALIGN while unaligned and HIT while aligned, and random selects either goal
uniformly. This tests symbolic goal advice rather than an LLM interpreting images.

`ALIGN_WITHOUT_FIRE` succeeds when the slots match after an action without
shooting; its reward is +1 for success, -1 for shooting, otherwise -0.01.
`HIT_TARGET` succeeds on the environment's target-hit event; its reward is +1 for
a hit, -0.1 for a missed shot, otherwise -0.01. Each conditioned replay transition
is labeled for both goals, yielding 64 training rows per sampled batch versus 32
for no-goal. The no-goal learner uses environment reward: -0.01 per decision,
an additional +1 for a hit or -0.02 for a missed shot. Double-DQN bootstrapping
stops on environment termination or requested-goal success; time-limit truncation
still bootstraps. No-goal bootstrapping stops only on environment termination.

Frozen-adviser training alternates behavior goals at reset or goal success. Both
policies receive the same planned training reset schedule per seed, starting at
100000, 100512, and 101024 respectively; actual resets differ as policies end
episodes at different times. The collection-teacher experiment instead trains a
separate policy for each teacher, with reset schedules starting at its training
seed. Saved reports contain the actual resets and replay coverage.

Hit rate counts successful episodes over all scheduled episodes. Mean decision
cost averages decisions to hit, assigning 300 to every failure, including adviser
errors. `decisions_to_hit_mean` averages successful episodes only; environment
return sums environment rewards, independent of relabeled goal rewards. Hits
follow [BasicV1Env](quickdraw_vizdoom/envs/basic.py): shooting at the target slot
or a native-engine kill can trigger success. These results therefore describe
this toy wrapper's mixed symbolic/native task.

The recorded LLM evaluation used OpenRouter `qwen/qwen3.5-9b`, temperature 0,
64 maximum output tokens, reasoning disabled, JSON output, and a 10-second request
timeout. The fixed prompt states the two goal definitions and the overall
target-hit objective. Requests, raw replies, returned model IDs, usage and timing
are in the report's `llm_prompt` and `teacher_audit` fields. The
[teacher adapter](quickdraw_vizdoom/goal_teachers.py),
[learner](quickdraw_vizdoom/learning.py), and
[evaluation runner](quickdraw_vizdoom/adviser_evaluation.py) implement this protocol.

## Current pilot findings

The completed [four-arm pilot](artifacts/quickdraw/adviser-four-arm-pilot-20261001-v2.json)
used 512 training decisions (481 updates) per policy and seed, with 30 evaluation
episodes per arm:

| Arm | Target hits | Hit rate | Mean decision cost (failure = 300) |
| --- | --- | --- | --- |
| No goal | 19/30 | 63.3% | 113.60 |
| Random goals | 7/30 | 23.3% | 230.87 |
| Rule goals | 5/30 | 16.7% | 251.27 |
| Qwen3.5-9B LLM goals | 11/30 | 36.7% | 191.37 |
| Always HIT_TARGET (post-hoc control) | 11/30 | 36.7% | 191.37 |

There were no adviser errors or evaluation updates, and all parameter hashes
remained unchanged. The three original baselines reproduced every episode record
from the [baseline pilot](artifacts/quickdraw/adviser-baseline-pilot-20261001-v2.json).
Qwen returned 30 valid goal choices, all `HIT_TARGET`, at a provider-reported total
cost of $0.0004341 and 55.47 seconds of cumulative request latency.

The observed LLM hit rate exceeded the current rule adviser by 20 percentage
points and random goals by 13.3 points. However, the
[post-hoc constant-goal control](artifacts/quickdraw/adviser-always-hit-check-20261001-v1.json)
matched all 30 LLM episode records, including outcomes, decision counts and goal
traces, using no API calls. Thus the observed gain is reproducible with a fixed
goal and does not demonstrate an advantage that requires an LLM. This control was
added after observing Qwen's choices; its selector latency was not measured.

Hits varied substantially across training seeds (ten episodes per cell):

| Training seed | No goal | Random | Rule | LLM / always HIT_TARGET |
| --- | --- | --- | --- | --- |
| 32001 | 10/10 | 0/10 | 0/10 | 0/10 |
| 34001 | 7/10 | 6/10 | 4/10 | 9/10 |
| 35001 | 2/10 | 1/10 | 1/10 | 2/10 |

Five of the six extra LLM hits over rule came from seed 34001. The goal policy for
seed 32001 hit zero targets with every adviser, demonstrating limited target-hit
execution at this budget. No-goal had the highest pooled hit rate, but its reward
and replay setup also differ. Thirty rollouts reuse three independently trained
policies per policy type and ten reset seeds; they are not thirty independent
training runs. These are small Basic-task findings, without a general claim about
LLM teaching or statistical significance. The v1 four-arm artifact retains an
earlier credential failure; it contains no model responses.

The earlier [collection-teacher pilot](artifacts/quickdraw/goal-teacher-pilot-20261001-v1.json)
also provides evidence that the learner responds to its goal input:

| Collection teacher | Correct goal success | Swapped goal success |
| --- | --- | --- |
| Rule | 37/60 (61.7%) | 8/60 (13.3%) |
| Random | 34/60 (56.7%) | 9/60 (15.0%) |

Each column covers both requested goals, three training seeds, and ten reset
seeds. Evaluation supplies the requested goal or its opposite while checking
success against the original request, stopping on that success or environment
end. The large success drop with swapped inputs supports goal-sensitive behavior.
The five-point rule/random gap is exploratory. No LLM was evaluated in this
collection pilot, and its combined alignment/hit success metric differs from the
full target-hit task above.

## Acknowledgements and citations

This work builds on ViZDoom, created by Michał Kempka, Marek Wydmuch, Grzegorz Runc,
Jakub Toczek, and Wojciech Jaśkowski. We thank the original authors, the
[Farama Foundation and ViZDoom contributors](https://github.com/Farama-Foundation/ViZDoom),
and the [ZDoom project](https://zdoom.org/) for the platform and engine underlying
these experiments. The QuickDraw task definitions and findings belong to this
research fork; the underlying ViZDoom platform is credited to its original authors
and contributors.

When reporting research using this fork, please cite the original ViZDoom papers:

1. Michał Kempka, Marek Wydmuch, Grzegorz Runc, Jakub Toczek, and Wojciech Jaśkowski
   (2016). [*ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement
   Learning*](https://arxiv.org/abs/1605.02097). IEEE Conference on Computational
   Intelligence and Games, pp. 341-348.
   [DOI: 10.1109/CIG.2016.7860433](https://doi.org/10.1109/CIG.2016.7860433).
2. Marek Wydmuch, Michał Kempka, and Wojciech Jaśkowski (2019).
   [*ViZDoom Competitions: Playing Doom from Pixels*](https://arxiv.org/abs/1809.03470).
   IEEE Transactions on Games, 11(3), pp. 248-259.
   [DOI: 10.1109/TG.2018.2877047](https://doi.org/10.1109/TG.2018.2877047).

The original [BibTeX entries](docs/upstream/ViZDoom-README.md#cite-as) and
[licensing information](docs/upstream/ViZDoom-README.md#license) are retained in the
archived README. Existing upstream copyright and licensing notices are preserved.
