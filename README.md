ANL 2024 Agent Skeleton
=======================

# Intelligent Negotiation Agent – Group 10

This project implements an autonomous negotiation agent ("Group10") developed using the NegMAS platform for the ANAC 2024 simulation.

## Description

The agent uses a dynamic strategy based on the SAO protocol. It adapts to different opponent types and negotiation phases, combining:

- Time-dependent aspiration functions
- Opponent modeling (concession, Nash-seeking, stubbornness)
- Pareto and Nash-based outcome selection
- Randomized fallback strategy for final steps

## Structure

- `group10.py` – Agent implementation
- `runner.py` – Run simulations against benchmark agents (A helper module to run a ournament with the agent)
- `group 10.pdf` – Strategy description and evaluation

==================

