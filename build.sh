#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser binaries (specifically chromium since it's the standard for scraping)
# Note: On Replit, this is handled via replit.nix (playwright-driver.browsers)
# playwright install chromium
