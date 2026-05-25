# Contributing to SLATE

Thank you for your interest in SLATE. Contributions are welcome from the research and engineering community.

## How to contribute

- **Bug reports and questions**: open an issue at https://github.com/physivitis/slate/issues
- **Code contributions**: open a pull request. For substantial changes, please open an issue first to discuss.
- **Validation contributions**: if you reproduce SLATE results at production scale (Pythia-410M or larger), please report your findings as an issue. We particularly welcome validation of Innovations 1 and 2 which are currently theoretically specified.

## Areas where contributions are most useful

- Implementation of Innovation 2 (Fisher-weighted token selection) within the NSA selection logic
- Validation at production scale (Pythia, larger models)
- Validation under stressed training conditions (the Lyapunov schedule needs hard conditions to demonstrate advantage)
- Hardware-specific optimisations of the spectral constraint enforcement
- Documentation improvements and example notebooks

## Code style

- PyTorch idiomatic Python 3.10+
- Type hints encouraged where they aid clarity
- Reasonable test coverage for new functionality

## License

Contributions are accepted under the MIT licence used by the project.
