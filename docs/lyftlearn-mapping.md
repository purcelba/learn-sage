# What this project maps to at Lyft — and where it doesn't

*Written after building a pCTR model end-to-end on SageMaker (see `PHASES.md`,
`NOTES.md`). Total cost: $0.02. Purpose: convert hands-on AWS mechanics into
vocabulary for the ads-ranking role, and — more usefully — into a list of
questions worth asking early.*

**Caveat on sourcing.** What I know about LyftLearn here comes from my own notes
before starting, not from the platform itself. Treat the Lyft column as a
hypothesis to correct, not a description. The AWS column is verified — I ran all
of it.

---

## The mapping at a glance

| What I built | Lyft equivalent | Fidelity |
|---|---|---|
| SageMaker Training Job running my `train.py` | **LyftLearn Compute** — training | **Close** |
| Batch Transform scoring a held-out file | **LyftLearn Compute** — scheduled batch scoring | **Close** |
| Model Registry: versioned artifact + approval status | model-artifact-gated-by-eval in CI/CD | **Partial** |
| SageMaker real-time endpoint | *nothing* — LyftLearn Serving is Kubernetes | **Breaks** |
| `features.py` shared by training and serving | a feature store | **Toy version** |
| `rank_candidates.py` | the auction/ad server | Deliberately outside the model |

---

## What maps cleanly: the offline half

**The training job is the real thing.** SageMaker provisioned an instance,
pulled a container, copied my code in, ran it, captured `/opt/ml/model` to S3,
and terminated. I wrote every line that affects predictions; the platform wrote
none of them.

That division is the whole point, and it's the LyftLearn Compute analogue:
**the platform runs it, the team owns what runs.** It became concrete when
things broke. Three failures got the first job to `Completed`: a zero training
quota, an execution role that couldn't read the bucket, and SageMaker rendering
a single-character hyperparameter as `-C` instead of `--C`. **All three were
platform-boundary problems. None touched the model.** Knowing which side of that
line a failure sits on is the thing that tells me whether to escalate to Ravi's
team or to my own.

**Batch Transform is the closer analogue of the two serving phases**, which is
worth stating plainly because it's counterintuitive: the *optional* phase is the
realistic one. Scheduled batch scoring is LyftLearn Compute's job. It also
proved something architectural — it reused `inference.py` **unchanged**, which
means the serving code never picked up an endpoint-specific assumption. The
result of that phase wasn't the predictions file; it was an empty `git diff`.

---

## What maps partially: the registry

I registered the model with an eval report attached and an approval status. That
*looks* like "versioned artifact gated by an eval report."

**It isn't a gate.** I asserted the registry "validates nothing," then tested it,
and was wrong in an interesting way:

| Checked at registration? | |
|---|---|
| Model artifact exists in S3 | **yes** — rejects if missing |
| Metrics file exists | **no** — a nonexistent path registers fine |
| The metric values | **no** — nothing opens the file |
| Approval justified by anything | **no** — I set `Approved` with one API call |

So a registry verifies the artifact is *present*, and nothing about whether it's
*good*. The actual gate is four separate things:

1. run the evaluation — **you**
2. decide whether it passes — **you**
3. record artifact, metrics location, verdict — *the registry*
4. refuse to deploy anything not `Approved` — **your pipeline**

Miss #4 and "Approved" means somebody set a string. Automating #2→#4 is a
different piece of machinery (SageMaker Pipelines' `ConditionStep`, or whatever
Lyft's CI equivalent is).

**This is the distinction I care most about carrying into the role:** "we have a
model registry" and "we have an enforced quality gate" are different claims, and
they look identical from a dashboard.

---

## Where it breaks down: serving

**The SageMaker endpoint maps to nothing at Lyft.** LyftLearn Serving runs on
Kubernetes. I built the endpoint anyway, to feel why that choice gets made.

What it costs: a dedicated `ml.t2.medium` at ~$0.065/hr — **~$1.56/day, ~$47/mo**
— billing from `InService` until deleted, regardless of traffic. I sent about 30
requests. The per-hour cost would have been identical had I sent zero.

Three ways that scales badly:

- **Cost is per model, not per request.** 50 models = 50 always-on instances,
  each sized for its own peak, each idle most of the time.
- **Utilisation is invisible.** Nothing in the bill distinguishes a saturated
  endpoint from an idle one.
- **You pay for peak, continuously.** Size for the busy hour, overpay for the
  other 23.

Kubernetes changes the arithmetic by bin-packing many models onto shared nodes,
scaling replicas with load and nodes with demand, and giving every model one
shared platform for rollouts, canaries, mTLS, tracing, and on-call.

What SageMaker buys instead is *no cluster to operate*. For one model, or a team
without platform engineers, that's a genuinely good trade. At Lyft's scale, with
a platform team already running Kubernetes, it inverts.

**What still transfers across the break:** the `inference.py` contract shape
(load model → parse request → featurize → predict → serialize) is
platform-independent, as is feature parity, as is keeping the model bid-agnostic.
What doesn't transfer: deployment mechanics, scaling, and the cost model.

---

## The parity discipline, which transfers completely

`features.py` is imported by both `train.py` and `inference.py`. Neither
reimplements it. Serving loads a `feature_config.json` saved at training time and
**refuses to start** if it disagrees — a dead endpoint being much better than one
quietly scoring a feature space the model never saw.

Measured result: endpoint predictions matched local scoring to **5.6e-17**, and
batch matched to the same, across 10,000 rows — despite different Python and
numpy versions in the container. That's not luck; it follows from there being
exactly one implementation.

At Lyft that guarantee is a feature store's job, and the scale of the problem is
different: a Spark job and an online store must agree, with point-in-time
correctness for training. The strongest version — used by most large ad systems —
is to **log features as served**, so training data is definitionally what serving
saw and skew becomes structurally impossible rather than something you test for.

One thing I got wrong worth keeping: I dropped the `id` column in data prep
because it isn't a feature. Correct for modelling, wrong for the pipeline —
batch scoring then produced 10,000 predictions with no key to join them back to
anything. **"Not a feature" and "not needed downstream" are different claims.**
The production shape keeps the identifier in the record, withholds it from the
model, and re-attaches it to the output.

---

## What this project does *not* cover

Worth being explicit, so I don't overclaim:

- **The retrieval/ranking funnel.** My model is the ranking stage only. Real ads
  systems cut millions of candidates to hundreds with a cheap decomposable model
  (two-tower, ANN over precomputed embeddings) before anything expensive runs.
- **A real feature store.** Every feature arrived in the request payload. In
  production, user features are *fetched* from an online store, computed offline.
- **Online monitoring.** No drift detection, no live calibration tracking, no
  alerting on prediction distribution shift.
- **Experimentation.** No A/B framework, no interleaving, no way to tell whether
  a model change actually improved anything.
- **Scale.** 50k rows and one instance. Nothing here exercises distributed
  training, sharding, or throughput.
- **Cross features.** Logistic regression on concatenated groups can only find
  linear relationships. Interactions are where ranking accuracy actually lives.

Incidentally: because my model is linear, it's additively separable — its score
genuinely *could* be decomposed into per-user and per-ad partials computed
offline. That's why early ad systems were linear. Adding cross features or a
hidden layer destroys that property, and that trade — give up cheap
precomputation to buy interactions — is the central architectural choice in
modern ranking.

---

## Questions for Ravi's team

The most useful output of this exercise. Each comes from something I hit.

**On the quality gate**
1. Where does enforcement actually live — an automated threshold in CI, or a
   human reading a dashboard before approving?
2. What happens if a model regresses on a key metric? Blocked automatically, or
   caught in review?

**On feature parity**
3. Are training features **logged as served**, or recomputed from raw logs later?
4. How would we *detect* training/serving skew in production today — is there a
   check, or would it show up as an unexplained metric drift?

**On the ranking system**
5. Where's the retrieval/ranking split, and what's the actual latency budget for
   the ranking stage?
6. What's precomputed offline vs. fetched vs. computed per request?

**On the platform boundary**
7. Who owns the training image and its dependency updates — platform or my team?
   (My scikit-learn version had to match the container exactly; an older runtime
   physically cannot load a newer pickle.)
8. When a training job fails, what's the triage path for "platform problem" vs.
   "our code problem"?

**On what my team owns**
9. Which of these is mine: feature definitions, model code, eval thresholds,
   deploy decision, on-call for serving?

---

## The thing I'd actually take away

Four times in this project, a green result meant nothing until I made something
fail:

- An IAM scoping test passed identically whether or not the scoping worked. A
  negative test showed two permissions had survived.
- I claimed the model registry validates nothing; testing showed it validates
  one thing and not three others.
- A batch job "succeeded" — but the meaningful result was an empty `git diff`,
  not the output file.
- A teardown script printed `CLEAN` while a stale resource sat in the same
  output, because cleanup was accidentally gated behind a billing check.

Four different services, one failure mode: **the happy path passing is not
evidence that the mechanism works.** For an IC that's a testing habit. For
someone running a team it's the difference between a dashboard that says the
gates are green and a system where the gates actually hold — and I'd rather find
out which one we have by asking early than by shipping a bad model.
