# Security

Treat character projects, Python scripts, Unreal plugins and downloaded model
backends as executable or untrusted inputs. Open only projects and scripts you
trust. Python Remote Execution should be enabled only on a trusted local network
when using the Unreal bridge.

Do not put API keys, private models, licensed character data, customer scans,
or identifying support bundles in issues or pull requests. Review and redact
logs before sharing. The source packager uses an explicit inventory and checks
for local paths and selected credential patterns; it is not a comprehensive
security scanner.

For a vulnerability, use the repository's private **Security > Report a
vulnerability** feature if its owner has enabled it. If unavailable, open an
issue requesting a private reporting channel without including exploit details
or sensitive files. Maintainers should enable private vulnerability reporting
and GitHub secret scanning before the public launch.

This preview has no guaranteed support window or response time. Reports should
include the affected version, a minimal reproduction, and the expected impact.
