#!/bin/bash
set -e

DOC_BRANCH="gh-pages"
GITHUB_REPO="https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/Pallas-Trace/pallas.git"
TMP_DIR=$(mktemp -d)

echo "📂 Cloning GitHub repository (branch $DOC_BRANCH) into $TMP_DIR"
if ! git clone --depth 1 --branch $DOC_BRANCH $GITHUB_REPO "$TMP_DIR"; then
  echo "Branch $DOC_BRANCH does not exist, initializing..."
  git clone --depth 1 $GITHUB_REPO "$TMP_DIR"
  cd "$TMP_DIR"
  git checkout --orphan $DOC_BRANCH
else
  cd "$TMP_DIR"
fi

echo "🧹 Cleaning old documentation"
rm -rf "$TMP_DIR"/*

echo "📁 Copying documentation files"
cp -r "$CI_PROJECT_DIR/static/"* "$TMP_DIR"
cp "$CI_PROJECT_DIR/daux.json" "$TMP_DIR"
cp "$CI_PROJECT_DIR/composer.json" "$TMP_DIR"
cp "$CI_PROJECT_DIR/Makefile" "$TMP_DIR" 2>/dev/null || true

echo "GITHUB_USER = ${GITHUB_USER}"
echo "GITHUB_USER_EMAIL = ${GITHUB_USER_EMAIL}"

git config user.email "${GITHUB_USER_EMAIL}"
git config user.name "${GITHUB_USER}"

git add .
if git diff --cached --quiet; then
  echo "No changes in documentation, nothing to do."
else
  git commit -m "Auto update documentation via GitLab CI"
  git push -f origin $DOC_BRANCH
  echo "🚀 Documentation synchronized on GitHub ($DOC_BRANCH)"
fi
