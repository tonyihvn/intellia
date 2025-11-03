Security incident remediation

If you've found secrets (API keys, tokens, passwords) committed to this repository, follow these steps immediately:

1. Revoke and rotate the leaked keys
   - For each leaked API key (OpenAI, Groq, Google/Vertex, OpenRouter, etc.), go to the provider's console and revoke/rotate the key immediately.
   - Do not wait to remove the key from the repository; rotate it first to prevent active misuse.

2. Remove the secrets from the repository working tree
   - Replace secrets in files with environment-variable references or placeholders (e.g., use an example file `src/config/llm_config.example.json` instead of committing real keys).
   - Add sensitive files to `.gitignore` (this repository already has entries for `src/config/llm_config.json`, `.env`, and `src/__pycache__`).

3. Remove secrets from git history
   - Use `git filter-repo` (recommended) or the BFG Repo-Cleaner to remove the secret from all commits.

   Using git filter-repo (recommended):

   - Install:
     - On macOS: `brew install git-filter-repo`
     - On Linux: `pip install git-filter-repo`

   - Example (replace SECRET with the actual secret string):

```bash
# Make a backup first
git clone --mirror <repo-url> repo-mirror.git
cd repo-mirror.git
# Remove the secret (example: an API key)
git filter-repo --invert-paths --paths-glob 'src/config/llm_config.json' || true
# OR to replace a specific secret string in all commits:
# git filter-repo --replace-text replacements.txt
# where replacements.txt contains lines like:
# SECRET==>REDACTED

# Push the cleaned repo back
git remote add origin-clean <repo-url>
git push --force origin-clean --all
```

   Using BFG (easier for simple cases):

```bash
# Create a mirror clone
git clone --mirror <repo-url>
cd <repo>.git
# Use BFG to remove passwords
bfg --delete-files src/config/llm_config.json
# Or remove a secret string:
# bfg --replace-text replacements.txt
# Followed by:
git reflog expire --expire=now --all && git gc --prune=now --aggressive
# Push cleaned repo
git push --force
```

4. Inform collaborators
   - Notify team members and downstream users of the incident and actions taken. Rotated keys will require updates to deployment and CI secrets.

5. Rotate secrets in deployment/CI
   - Update environment variables in deployments, CI/CD, and any infrastructure that used the leaked keys.

6. Monitor for abuse
   - Check provider dashboards for unusual activity and set up alerts or usage limits.

Notes
- After rewriting history you will need to force-push and all collaborators will need to reclone the repository or reset their local clones.
- If you're unsure or need help, contact your security team.
