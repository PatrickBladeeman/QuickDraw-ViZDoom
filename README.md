# QuickDraw-ViZDoom

QuickDraw-ViZDoom is a research fork of [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)
for testing goal-conditioned reinforcement learning and LLM advisers. The current
Basic task compares no goal conditioning, random goals, rule-based goals, and
LLM-selected goals on held-out target-hit episodes.

For new comparisons, use the [native controls](#native-parity-and-hierarchy-controls),
which score actual Doom kills. The historical Basic pilots below use the
wrapper-defined hit metric.

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

| Arm | Wrapper target hits | Hit rate | Mean decision cost (failure = 300) |
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

### Diagnostic follow-up (2026-10-01)

The [environment audit](artifacts/quickdraw/gc-environment-audit-20261001-v1.json)
found that the symbolic target is `seed % 9 - 4`, while ACS independently spawns
the visible Doom target at a random native position. Moving to the symbolic slot
and shooting scored 18/18 wrapper hits with **zero native kills**. Across 100
reset seeds, 67 starting observations shared an exact image hash with a different
symbolic target slot. This demonstrates ambiguity in the initial visual input,
without proving that every observation history is uninformative. The reported
pilot metric is the wrapper's mixed symbolic/native success event, not a count
of native target kills.

The [saved-pilot diagnosis](artifacts/quickdraw/gc-saved-pilot-diagnosis-20261001-v1.json)
also isolates two execution failures: 14/27 initially unaligned rule episodes
never selected HIT, and only 4/13 ALIGN-to-HIT handoffs ended in a wrapper hit.
The three initially aligned episodes added one hit. Alternating goal selections
did not balance behavior time: seed 35001 spent 475/512 decisions under ALIGN,
although exhaustive relabeling still supplied 512 training rows for each goal.
The earlier collection-teacher rates describe separately trained policies.

[Complete representative traces](artifacts/quickdraw/gc-policy-trace-20261001-v2.json)
record actions, six Q values, slots, native kill counts and unchanged parameter
hashes. One policy moved right while its symbolic target was left; another trace
reported a native-kill fallback hit while the two symbolic slots disagreed.
The current goal vector chooses ALIGN or HIT and carries no explicit left/right
direction. Establish agreement between visible state, adviser state and rewards
before using larger runs to assess goal conditioning or LLM-specific benefit.

## Native parity and hierarchy controls

The [native environment](quickdraw_vizdoom/envs/native_basic.py) uses the visible
Cacodemon's actual coordinates, actual ammunition, and native `KILLCOUNT`.
Alignment means a lateral error within 8 world units; each decision advances four
native tics. Only a native kill awards a hit. The original Basic wrapper and
historical artifacts remain available for reproducing the earlier pilots.

The [native geometry check](artifacts/quickdraw/native-environment-check-20261001-v1.json)
uses a scripted controller with access to the actual lateral error. It killed the
native target on all 100 evaluation seeds, with zero false hit events. This checks
task reachability and scoring; it is not a learned-policy or LLM result.

Run the matched controls:

```powershell
.venv\Scripts\python.exe -m quickdraw_vizdoom.native_evaluation --output artifacts/quickdraw/native-controls-new.json
```

Defaults are 4,096 decisions per policy, five training seeds (32001, 33001, 34001,
35001, 37001), and all 100 held-out reset seeds. Four policies are trained per
seed: no-goal, fixed HIT, alternating goals, and rule-guided goals. No-goal and
fixed HIT use identical environment rewards and one training row per transition;
the two variable-goal policies use goal rewards and two relabeled rows. They use
the same architecture, optimizer, budget and planned reset schedule within each
seed. A constant one-hot goal can change initial Q values and optimization, so
the fixed-HIT comparison is a parity control, not evidence of useful goal context.

Seven frozen evaluation arms compare no-goal, fixed HIT, alternating-trained
HIT/rule/random, and rule-trained HIT/rule. Correct/swapped ALIGN and HIT diagnostics
use the first 20 held-out seeds per training seed; `--diagnostic-episodes 100`
expands them to the full schedule. Reports include native hit counts/rates,
environment returns, success-only latency, capped decision cost, handoff success,
behavior-goal selection and decision counts, sampled training rows, training time,
paired differences and unchanged parameter hashes. The first episode of each arm
also records actions, Q values, lateral errors and native kill counts.

The [completed native baseline](artifacts/quickdraw/native-controls-4096-five-seed-20261001-v2.json)
covers five training seeds and 500 episodes per arm. Decision cost counts the
actual decisions for hits and assigns 300 to failures.

| Training | Adviser | Native hits | Mean decision cost |
| --- | --- | --- | --- |
| No goal | None | 351/500 (70.2%) | 92.98 |
| Fixed HIT, environment reward | Constant HIT | 363/500 (72.6%) | 85.81 |
| Alternating goals | Constant HIT | 384/500 (76.8%) | 73.16 |
| Alternating goals | Rule | 375/500 (75.0%) | 80.05 |
| Alternating goals | Random | 327/500 (65.4%) | 108.41 |
| Rule-guided goals | Constant HIT | 333/500 (66.6%) | 105.05 |
| Rule-guided goals | Rule | 208/500 (41.6%) | 180.45 |

**Concrete finding:** changing only the adviser on frozen alternating-trained
weights gives rule advising a 9.6 percentage point hit-rate advantage over random
advising, but constant HIT scores another 1.8 points higher. On rule-trained
weights, constant HIT beats rule advising by 25.0 points. The rule-trained
hierarchy never completed initial ALIGN in 231/500 episodes and hit in only
203/264 completed handoffs. The alternating-trained hierarchy hit in 370/374
handoffs. Alignment and subsequent HIT execution can therefore be measured
separately on the coherent native task.

The correct/swapped diagnostic uses 100 episodes per cell (20 held-out reset
seeds per training seed):

| Training | Correct ALIGN | Swapped ALIGN | Correct HIT | Swapped HIT |
| --- | --- | --- | --- | --- |
| Alternating goals | 67/100 | 18/100 | 70/100 | 0/100 |
| Rule-guided goals | 50/100 | 14/100 | 64/100 | 0/100 |

The goal input affects behavior, while learning remains sensitive to the training
seed. No-goal hit rates range from 5% to 100%; fixed-HIT rates range from 26% to
100%. Their pooled 2.4-point difference does not establish a dependable benefit
from a constant goal. Comparisons with variable-goal training also change the
reward and relabeling budget, so they measure the complete training system.
Each policy made 4,096 training decisions and 4,065 optimizer updates; variable
goals sampled 260,160 training rows versus 130,080 for the parity controls.
All 3,500 primary and 800 diagnostic episodes passed native-hit, frozen-weight,
adviser-failure and infrastructure checks. These are exploratory results from
five training seeds, with zero LLM provider calls.

An LLM comparison reuses a native checkpoint with
`--arms alternating_hit alternating_rule alternating_random llm --checkpoint <path>`.
All four arms then share the same alternating-trained frozen policy. The existing
OpenRouter configuration applies; the native prompt uses actual alignment
geometry. Baseline runs make zero provider API calls. Outputs reject overwrites.

The [saved five-seed policies](artifacts/quickdraw/native-controls-4096-five-seed-20261001-v2-policies.pt)
were reused in the [completed two-model evaluation](artifacts/quickdraw/native-llm-comparison-five-seed-20261001-v2.json).
Both [Qwen3.5-9B](https://openrouter.ai/qwen/qwen3.5-9b) and
[DeepSeek V3.2](https://openrouter.ai/deepseek/deepseek-v3.2) received the same
native summary, prompt and goal-persistence protocol. Each ran all 100 evaluation
seeds for each of the five frozen alternating-trained policies. Temperature was
zero, reasoning disabled, output limited to 64 tokens, and each request had a
10-second deadline. Invalid JSON or request failures ended the episode as a
failure, with no rule fallback. No training was repeated; the validated baseline
episode rows were reused for paired comparisons.

| Adviser on shared alternating-trained weights | Native hits | Mean decision cost | Failed adviser episodes |
| --- | --- | --- | --- |
| Constant HIT | 384/500 (76.8%) | 73.16 | 0 |
| Rule | 375/500 (75.0%) | 80.05 | 0 |
| Random | 327/500 (65.4%) | 108.41 | 0 |
| Qwen3.5-9B | 370/500 (74.0%) | 82.49 | 12 |
| DeepSeek V3.2 | 375/500 (75.0%) | 80.05 | 0 |

**LLM finding:** DeepSeek reproduced the rule adviser exactly: all 500 goal
sequences and all 500 hit/decision/return outcomes matched. Qwen beat random
advising by 8.6 percentage points, but lost to rule advising by 1.0 point and
constant HIT by 2.8 points. These results do not establish an LLM-specific
advantage on the current two-goal task.

Qwen had eight timeouts and four malformed-JSON responses. In a secondary
analysis of the 488 episodes with no adviser error, it hit 370 targets versus
373 for constant HIT and 364 for rules on those exact same episodes. The primary
rates above retain all failures. Its initial choices also differed by direction:
all 144 valid negative-error requests selected HIT, while 239 positive-error
requests selected ALIGN. This is a descriptive choice pattern; the run does not
isolate its causal effect.

Qwen made 724 API requests, with reported cost of $0.01271 on 716 responses;
DeepSeek made 874, with reported cost of $0.02871 on all 874 responses. Mean
observed request latency was 1.04 and 1.84 seconds respectively. OpenRouter used
default provider routing and five concurrent evaluation workers. Costs cover
completed evaluation usage with reported billing, excluding preflight calls,
the interrupted baseline-replay attempt and unreported timeout billing. Model
responses may vary across calls even at temperature zero. All LLM episodes used
unchanged policy hashes and native kill scoring, with zero infrastructure failures.

To reproduce an evaluation with either model, set `OPENROUTER_API_KEY` in the
process environment and use a new output filename:

```powershell
.venv\Scripts\python.exe -m quickdraw_vizdoom.native_evaluation --checkpoint artifacts/quickdraw/native-controls-4096-five-seed-20261001-v2-policies.pt --arms llm --diagnostic-episodes 0 --llm-model qwen/qwen3.5-9b --output artifacts/quickdraw/native-qwen-new.json
```

Replace the model ID with `deepseek/deepseek-v3.2` for the second model. Pair
the resulting episodes with the saved baseline by training seed and reset seed.

## Instruction-conditioned adviser benchmark

The [instruction evaluator](quickdraw_vizdoom/instruction_evaluation.py) varies
the requested outcome while retaining native Basic and the existing ALIGN/HIT
skills. Instructions request alignment with no shots and a surviving target,
a native kill, or alignment before the first actual shot followed by a kill.
An adviser returns one of three bounded plans: ALIGN, HIT, or ALIGN then HIT.
Execution stops when the plan finishes. Task success checks actual ammunition
expenditure, alignment history and `KILLCOUNT`; it is not the earlier hit-rate
metric. Equivalent plans receive credit when their actual behavior obeys the
instruction. Exact canonical-plan matching is a secondary diagnostic.

The [20 development instructions](artifacts/quickdraw/instruction-development-v1.jsonl)
were used to implement a lexical rule parser with synonyms, negation and
ordering. It achieved 100% native task success with the scripted executor on
that development set. Another agent authored and froze the
[60 held-out instructions](artifacts/quickdraw/instruction-test-v1.jsonl)
without inspecting the parser. A separate
[60-instruction confirmation set](artifacts/quickdraw/instruction-confirmation-v1.jsonl)
was commissioned before examining the first outcomes. Each evaluation set has
ten distinct wording families and twenty instructions per task; families and
exact instructions do not overlap between sets or with development. These are
agent-authored synthetic cases with independently reviewed labels, not a
human-user instruction distribution.

The rule and Qwen advisers receive only the instruction and share the same
three available programs. Gold task labels never enter the LLM request.
Qwen3.5-9B uses the existing OpenRouter JSON adapter, temperature zero, reasoning
disabled, 64 output tokens and a ten-second deadline. Each instruction receives
three independent requests. Invalid output or transport errors count as failed
plans with task decision cost 300; there is no fallback. Rule abstentions also
count as failures. Neither prompt nor rule grammar changed after testing began.

Rule, LLM, constant-HIT, random-plan and gold canonical-plan controls execute on
the same five saved alternating-trained GC policies. The no-goal control uses
the five paired unconditioned policies. All received 4,096 training decisions;
the no-goal learner used environment reward and one row per transition, while
GC used goal rewards and two relabeled rows. Thus LLM versus rule isolates the
adviser on shared weights, whereas versus no-goal is an end-to-end system
comparison. Evaluation performs zero updates and verifies parameter hashes.

The scripted geometry executor separately diagnoses instruction interpretation
and task reachability. Its HIT implementation aligns before firing; it is not
a direct-fire learner. Deterministic execution is cached once per
policy/program/reset seed. The resulting instruction scores are counterfactual
re-scoring of those trajectories, not independent new environment episodes.
Reports retain the native action traces and report physical episodes and
decisions separately. Confidence intervals resample wording families, keeping
requests, layouts and policies within each family. They are conditional on the
tested frozen policies and layouts, not uncertainty across future training runs.

The [first held-out pilot](artifacts/quickdraw/instruction-qwen-heldout-20261001-v1.json)
and [independent wording confirmation](artifacts/quickdraw/instruction-qwen-confirmation-20261001-v1.json)
each used ten held-out reset seeds and all five policy pairs:

| Adviser/control | First set task success | Confirmation task success |
| --- | --- | --- |
| Qwen3.5-9B | 79.40% | 77.13% |
| Frozen lexical rule | 62.57% | 56.73% |
| No-goal learner | 24.67% | 24.67% |
| Constant HIT on GC | 30.00% | 30.00% |
| Random plan on GC | 36.24% | 36.24% |
| Gold canonical plan on GC | 79.33% | 79.33% |

Qwen's advantage over rules was +16.83 percentage points on the first set
(97.5% paired family-bootstrap interval: +7.93 to +25.97) and +20.40 on
confirmation (+8.07 to +30.77). The two primary comparisons are LLM versus rule
and no-goal; each uses a 97.5% interval. Each pilot actually executed 230 native
episodes and 14,263 decisions. Each learned arm contains 9,000 counterfactual
task scores; the uncertainty unit is ten wording families, not those scores.
Qwen failed on 0/180 first-set requests and 2/180 confirmation requests. Rules
abstained on eight and nine unique instructions respectively.

The [expanded confirmation](artifacts/quickdraw/instruction-qwen-confirmation-full-20261001-v1.json)
uses the same confirmation instructions, parser, prompt and five frozen policy
pairs across **all 100 held-out reset seeds**. It makes a fresh set of 180 Qwen
requests, retaining two failed requests in the primary rates:

| Adviser/control | Constraint-satisfying task success | Mean task decision cost |
| --- | --- | --- |
| Qwen3.5-9B | **73.76%** | **82.59** |
| Frozen lexical rule | 54.41% | 139.53 |
| No-goal learner | 24.00% | 229.42 |
| Constant HIT on GC | 29.87% | 211.76 |
| Random plan on GC | 34.73% | 197.74 |
| Gold canonical plan on GC | 75.87% | 76.37 |

Qwen improves task success over the shared-policy rule adviser by **19.35
percentage points**, with a 97.5% paired family-bootstrap interval of **+7.57
to +29.49**. Its end-to-end difference from no-goal is +49.76 points
(+45.97 to +51.87). The difference from rules is positive for each of the five
training seeds, although one wording family favors rules. The actual execution
budget is 2,300 native episodes and 162,772 decisions; each learned arm has
90,000 counterfactual task scores, which are not independent observations.
The scripted gold-plan executor satisfies every task on all 100 layouts.
No policy updates, parameter changes or infrastructure failures occurred.
Rules abstained on nine of sixty unique instructions. Reported billing for this
expanded run is $0.00313 across 178 responses, excluding unreported timeout
billing; total request latency is 199.44 seconds across 180 calls.

**Supported scope:** Qwen generalizes these held-out instructions better than
this development-frozen lexical parser, and that difference improves completion
through shared learned skills. Much of the rule deficit comes from missing
paraphrases. A post-hoc diagnostic restricted to the 51 confirmation instructions
that rules could parse still gives Qwen 73.78% versus rules 64.02% across all
100 layouts; it retains LLM errors. The primary comparison above retains all
sixty instructions. A [compact findings summary](artifacts/quickdraw/instruction-findings-20261001-v1.json)
records all three runs, hashes, costs, primary comparisons and this diagnostic.
This does not establish superiority over every rule system or an
intrinsic need for LLM tactical reasoning. The gold-plan control also exposes
remaining low-level execution failures. Language-directed composition of
existing grounded skills follows the approach explored in
[SayCan (Ahn et al., 2022)](https://arxiv.org/abs/2204.01691); our experiment is a
small synthetic instruction benchmark, not a reproduction of that study.

Reproduce with the existing checkpoint and a new output path:

```powershell
.venv\Scripts\python.exe -m quickdraw_vizdoom.instruction_evaluation --cases artifacts/quickdraw/instruction-confirmation-v1.jsonl --checkpoint artifacts/quickdraw/native-controls-4096-five-seed-20261001-v2-policies.pt --llm-model qwen/qwen3.5-9b --evaluation-episodes 100 --output artifacts/quickdraw/instruction-confirmation-new.json
```

Set `OPENROUTER_API_KEY` in the process environment for LLM requests. Omit
`--llm-model` to run controls without provider calls, or omit `--checkpoint` for
the scripted diagnostic alone. Reports reject overwrites and record source,
case and checkpoint hashes before requests. Future parser/prompt tuning needs
new held-out wording families; preserve these results.

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
