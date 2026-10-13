# Security Policy

## Overview

Leaf is an open-source, lightweight CPU inference framework built with C++ and Python.

We take security reports seriously and welcome responsible vulnerability disclosures that help protect Leaf and its users.

Leaf is currently under active development. Security improvements, vulnerability fixes, and compatibility changes may occur as the project evolves.

## Supported Versions

| Version | Security Support |
|---|---|
| `main` (active development) | Best-effort security fixes |
| Older commits and experimental builds | Not guaranteed |
| Future tagged releases | Support status will be documented when released |

At present, Leaf does not guarantee long-term security maintenance or backports for older versions.

## Reporting a Vulnerability

**Please do not disclose security vulnerabilities through public GitHub issues, pull requests, or discussions before coordinated disclosure.**

To report a vulnerability:

1. Navigate to the [Leaf Security page](https://github.com/Nbit-51/Leaf/security).
2. Select **Report a vulnerability** if GitHub Private Vulnerability Reporting is enabled.
3. Submit a private report describing the problem and its potential impact.

If private reporting is unavailable, open a public issue requesting a private security contact method, without including vulnerability details or reproduction instructions.

A useful security report includes:

- A description of the vulnerability
- The affected Leaf version or Git commit
- The affected component or file
- Operating system, CPU architecture, and relevant runtime details
- Steps to reproduce the issue, using sanitized data
- Potential security impact
- Suggested mitigation, if available

Please avoid including credentials, personal information, private model weights, or sensitive datasets in reports.

## Security Scope

We particularly welcome reports concerning:

- Memory-safety vulnerabilities in native C++ kernels
- Buffer overflows, out-of-bounds access, and integer overflows
- Unsafe parsing of model weights, metadata, or serialized artifacts
- Malformed tensor shapes and untrusted input dimensions
- Memory-mapped artifact handling
- Unsafe file-system and cache operations
- Dependency and software supply-chain vulnerabilities
- Accidental exposure of sensitive files or credentials
- Security weaknesses in model import and execution paths

Ordinary correctness bugs, unsupported architectures, and performance regressions should generally be reported through GitHub Issues unless they have a security impact.

## Disclosure and Response Process

After receiving a report, maintainers will aim to:

1. Acknowledge and review the report as availability permits.
2. Investigate whether the issue is reproducible.
3. Assess severity, affected configurations, and potential impact.
4. Develop and validate an appropriate fix.
5. Coordinate disclosure and publish a security advisory when warranted.

There is currently no guaranteed response or remediation timeframe.

We appreciate reporters coordinating with maintainers before public disclosure to allow reasonable time for investigation and remediation.

## Security Best Practices for Users

Because Leaf executes native code and processes model artifacts, users should:

- Obtain model files from trusted sources.
- Keep Leaf and its dependencies updated.
- Avoid running untrusted artifacts in privileged environments.
- Review third-party scripts and model preparation tools before execution.
- Avoid committing API tokens, private datasets, or sensitive model credentials.
- Treat experimental kernels and unverified optimizations as development features.

Leaf is an early-stage project and should not be assumed to have undergone a comprehensive security audit.

## Security Research and Recognition

Constructive security research and responsible disclosure are welcome.

With the reporter's permission, confirmed findings may be acknowledged in release notes or security advisories.

Leaf does not currently operate a paid bug bounty program.

## License

Leaf is distributed under the Apache License 2.0.

See [LICENSE](https://github.com/Nbit-51/Leaf/blob/main/LICENSE) for the applicable license terms.

Thank you for helping make Leaf safer and more reliable.
