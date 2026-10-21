# Optional recheck of the composed Risk-MPC plan

Clearance of sampled plans does not imply clearance of their weighted control
sequence. The composition describes a different trajectory and must be rolled
out separately if its predicted clearance is to be checked.

## Reproduce the counterexample

Run `python tests/test_composed_plan.py` with the project dependencies installed.
The fixture uses the native analytic unicycle rollout and native trajectory cost;
it supplies fixed action samples and obstacle futures instead of trained models.
The production recheck uses the planner's configured robot-model rollout.

At baseline commit `cfcb272`, place the robot at `(5, 10, 0)` and a stationary
obstacle at `(6.4, 10)`. Two constant action sequences `(1.2, +1.5)` and
`(1.2, -1.5)` over 18 steps have equal native costs and minimum predicted
clearance approximately `0.7619 m`. Their equally weighted mean `(1.2, 0)`
has clearance `0.04 m`, below the native collision distance `0.7 m`.
This is a predicted-horizon counterexample, not a closed-loop collision statistic.

## Enable the proposed behavior

```python
planner = RiskAwareNeuralMPC(
    robot_model, ensemble_model, recheck_composed_plan=True
)
```

The default is `False`, preserving the original sampling, cost, weighting and
warm-start behavior. When enabled, the planner rolls out the weighted plan with
the same robot model used to score samples. If any supplied future predicts a
collision, it chooses the finite-cost sample with lowest CVaR among samples that
clear every supplied future. The selected plan is shifted for the next call in
the existing way. The tests cover all-future eligibility, cost selection, clear
mean preservation, warm-start shifting, nonfinite predictions and the threshold.

If no eligible sample exists, the baseline weighted plan is retained. It can
still collide. This feature supplies no guarantee in that case. It also cannot
establish physical robot safety: model error, unrepresented futures, collisions
between time steps, walls and tracking error remain outside this recheck.
The clearance check uses `distance >= SAFE`, matching the native cost's strict
`distance < SAFE` collision rule. It is not an additional safety margin.

## Evaluation still needed

Local verification (Python 3.12; NumPy 2.3.5, SciPy 1.17.0,
scikit-learn 1.8.0, Matplotlib 3.10.8): all 10 new tests pass, along with
`python tests/smoke_test.py` and `python tests/test_upgrade_regression.py`.
Running the same new suite against unchanged `cfcb272` gives 4 assertion
failures and 2 errors for the then-absent clearance helper. The native
counterexample itself passes on both versions, documenting the baseline issue.
These local dependency versions differ from the CPU constraints; the configured
Python 3.10/3.12 and Docker CI checks have not been run locally.

Compare default and enabled planners with identical scenarios, model checkpoints
and seeds. Report collision and success rates, control effort and planning
latency. A mean rollout is added on each enabled call; sample clearance is
checked only if that composed rollout fails. The deterministic tests do not
measure the resulting runtime or establish improved closed-loop outcomes.

Conceptual analysis: Massimiliano Brighindi (M). Patch and tests prepared with
an AI assistant for review by the project maintainer.
