# Agent Instructions

## GitHub Actions — Pin to commit SHA

Always pin third-party GitHub Actions to a full-length commit SHA, not a
version tag or branch reference.  This prevents supply-chain attacks where
a tag is moved to a malicious commit.

The SHA must be followed by a `# <tag>` comment so humans can quickly
identify the intended version during code review.

**Example:**
```yaml
uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5  # v4
```

When upgrading an action, look up the SHA for the new tag with:
```bash
git ls-remote https://github.com/<owner>/<repo>.git refs/tags/<tag>
```
