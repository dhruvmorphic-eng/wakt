# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.4.x | Yes |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly.

**Do not open a public issue.**

Instead, email **dhruvcp07@gmail.com** with:

1. A description of the vulnerability
2. Steps to reproduce it
3. The potential impact
4. Any suggested fix (optional)

You will receive an acknowledgment within 48 hours. We will work with you to
understand the issue and coordinate a fix before any public disclosure.

## Scope

Wakt runs entirely on-device — there is no hosted service. Security concerns
typically involve:

- Model input handling (adversarial inputs, prompt injection into state text)
- The optional HTTP server (`wakt-serve`) when exposed to a network
- Dependency vulnerabilities in PyTorch, Transformers, or optional extras
