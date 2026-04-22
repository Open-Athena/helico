---
name: Experiment
about: Propose a research experiment. The notebook lives on a branch; results come back here.
title: "exp: "
labels: ["experiment"]
assignees: []
---

<!--
This template is the preregistration for an experiment. Fill out every
section. An experiment is a jupytext-markdown notebook at
experiments/exp<issue#>_<slug>/README.md on a branch; when you run it, it
dispatches Modal training/bench jobs and produces plots + HTML.

See experiments/TEMPLATE.md for the notebook skeleton and
.agents/project/20260422_experiment_system_design.md for the full workflow.
-->

## Question

<!-- What do you want to learn? One sentence. -->

## Hypothesis

<!-- What do you expect to see and why? This is what we're preregistering. -->

## Background

<!-- Prior runs, papers, conversations this builds on. Link issues/PRs. -->

## Approach

<!-- Outline what the notebook will do. Bullet points are fine. E.g.:
     - Run FoldBench on the Protenix v1 checkpoint
     - Score across all 9 categories
     - Compare monomer LDDT distribution to published Protenix numbers
-->

## Compute estimate

<!-- Rough GPU type, count, and wall-clock hours. Used for pre-run cost check. -->

- GPU: <!-- e.g. H100 -->
- GPU count / workers: <!-- e.g. 8 -->
- Estimated wall hours: <!-- e.g. 0.5 -->
- Type: <!-- `training`, `bench_only`, or `mixed` -->

## Success criteria

<!-- How will we know if the hypothesis held? Concrete metrics + thresholds.
     Example:
       - interface_protein_protein.mean_lddt improves by >= 0.02 vs baseline
       - monomer_protein.mean_lddt does not regress
-->

## Baselines

<!-- Named prior runs or published numbers to compare against. -->

## Notes

<!-- Anything else the agent or a reviewer should know. -->
