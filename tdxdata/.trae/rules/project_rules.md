# Project Rules

## 1. Git Configuration
- Credentials are stored in `~/.git-credentials` (configured via `git config credential.helper store`).
- Do NOT store tokens or secrets in repository files.
- Git username: `peiking88`.
- Access GitHub repositories using `github.com` domain (e.g., `https://github.com/peiking88/...`).
- Disable SSL certificate verification on push: `git config http.sslVerify false`.
- Ensure no sensitive information in commits (push without SSL verification).

## 2. Language
- Use Chinese for all conversational output during work.
- Use English for commit messages and README files.

## 3. Parallel Compilation
- Always use `-j$(nproc)` parameter when running `ninja` or `make` for parallel compilation.

## 4. Third-party Code Protection
- NEVER modify source files under `3rdparty/` or `external/` directories.

## 5. Unit Testing
- All unit tests MUST be executed correctly. NEVER skip any test.
- All unit tests MUST pass before starting any new task.

## 6. Debugging
- NEVER simplify or bypass issues during debugging. Always investigate and fix root causes.

## 7. Log Management
- All log files MUST be saved under the `log/` directory.

## 8. Configuration
- Configuration files MUST be placed under the `cfg/` directory.

## 9. Source Files
- Source code MUST be organized under the `src/` directory.

## 10. Documentation
- Documentation files MUST be placed under the `docs/` directory.

## 11. Scripts
- Script files MUST be placed under the `scripts/` directory.

## 12. Testing Architecture
- Support both real and mock tests in the architecture.
- Real tests should be prioritized over mock tests when both are available (real > mock).

## 13. Test Coverage
- Third-party components (external dependencies) are NOT included in unit test coverage.
- Do NOT write unit tests for third-party components.

## 14. Phase Completion
- All unit tests MUST pass before marking a phase as complete.

## 15. Process Control
- Use TodoWrite tool to create and track project task lists.
- Track progress through task states (pending -> in_progress -> completed).

## 16. Full Completion
- ALL tests (unit + integration) MUST pass before marking project as fully complete.
