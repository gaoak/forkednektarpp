# Instructions for AI coding agents

Before writing or changing any code in this repository, read and follow:

- [CONTRIBUTING.md](CONTRIBUTING.md) — contribution process, commit and
  merge-request conventions, formatting requirements.
- [docs/agents/CodingPractices.md](docs/agents/CodingPractices.md) — the
  coding conventions agreed in redesign code review. These are binding on
  new and modified code in the redesign `Operators` library and its unit
  tests, and are the conventions review will hold a change to.

Where the two disagree, CONTRIBUTING.md takes precedence.

Match the surrounding code. Run `clang-format` (version 18, as CONTRIBUTING.md
requires) before committing; a CI job verifies it and will fail the merge
request otherwise. XML files should be formatted using `.gitlab-ci/formatXMLfile.sh path-to-xml-file path-to-xml-file`.
