#!/usr/bin/env bash
# Сборка Docker-образа Whisper STT и опциональный push в Docker Hub.
#
# Ожидаемые переменные окружения:
#   DOCKER_ORGANIZATION  — организация / namespace на Docker Hub
#   DOCKER_IMAGE_NAME    — имя репозитория образа
#   DOCKER_USERNAME      — логин Docker Hub (нужен при --push)
#   DOCKER_PASSWORD      — пароль / access token (нужен при --push)
#
# Примеры:
#   ./scripts/build-image.sh v1.2.3
#   ./scripts/build-image.sh --push v1.2.3
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PUSH=0
VERSION=""

usage() {
  echo "Usage: $0 [--push] <version>" >&2
  echo "  version — тег вида v1.2.3 (как GITHUB_REF_NAME при push tags)" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --push)
      PUSH=1
      shift
      ;;
    -h|--help)
      usage
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      ;;
    *)
      if [[ -n "$VERSION" ]]; then
        echo "Extra argument: $1" >&2
        usage
      fi
      VERSION="$1"
      shift
      ;;
  esac
done

if [[ -z "$VERSION" ]]; then
  usage
fi

: "${DOCKER_ORGANIZATION:?DOCKER_ORGANIZATION is required}"
: "${DOCKER_IMAGE_NAME:?DOCKER_IMAGE_NAME is required}"

IMAGE="${DOCKER_ORGANIZATION}/${DOCKER_IMAGE_NAME}"
TAG_VERSION="${IMAGE}:${VERSION}"
TAG_LATEST="${IMAGE}:latest"

echo "Building ${TAG_VERSION}"
docker build \
  --pull \
  -t "${TAG_VERSION}" \
  -t "${TAG_LATEST}" \
  .

if [[ "$PUSH" -eq 1 ]]; then
  : "${DOCKER_USERNAME:?DOCKER_USERNAME is required for --push}"
  : "${DOCKER_PASSWORD:?DOCKER_PASSWORD is required for --push}"

  echo "Logging in to Docker Hub as ${DOCKER_USERNAME}"
  echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin

  echo "Pushing ${TAG_VERSION}"
  docker push "${TAG_VERSION}"
  echo "Pushing ${TAG_LATEST}"
  docker push "${TAG_LATEST}"
fi

echo "Done: ${TAG_VERSION}"
