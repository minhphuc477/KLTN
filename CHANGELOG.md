# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Fix: remove redundant scheduler call in `ZeldaGUI._start_auto_solve` and make `_schedule_solver` atomic (add `_solver_lock`) to prevent duplicate solver scheduling; added unit tests for scheduling and concurrency. (PR: draft)
