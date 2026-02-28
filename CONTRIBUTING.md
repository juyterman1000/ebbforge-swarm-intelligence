# Contributing to OpenRustSwarm

First off, thank you for considering contributing to OpenRustSwarm! It's people like you that make OpenRustSwarm such a powerful tool.

## Technical Standards
- **Performance First**: All core logic must be implemented in Rust with a focus on zero-copy and cache-locality.
- **Safety**: No `unsafe` blocks without extensive justification and unit testing.
- **Reproducibility**: New features must include a verification script compatible with `verify_all_benchmarks.py`.

## How to Contribute
1. **Reporting Bugs**: Use GitHub Issues. Include your OS, Rust version, and a minimal reproducible example (MRE).
2. **Feature Requests**: Open an issue to discuss the architectural impact before submitting code.
3. **Pull Requests**:
   - Branch from `main`.
   - Ensure `cargo fmt` and `cargo test` pass.
   - Tag your PR with `enhancement`, `bug`, or `performance`.

## Development Setup
```bash
git clone https://github.com/juyterman1000/openrustswarm.git
cd openrustswarm/openrustswarm-core
cargo build --release
```

Thank you for being part of the swarm!
