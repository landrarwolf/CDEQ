#!/usr/bin/env bash
set -u

STAGE=/private/tmp/cllm-stage
LOG_FILE="${STAGE}/background.log"

if [[ -f "${STAGE}/download.done" ]]; then
  echo 'STATUS: COMPLETE - all remote artifacts verified'
elif screen -ls 2>/dev/null | grep -q '[.]cllm-artifacts'; then
  echo 'STATUS: RUNNING (screen session: cllm-artifacts)'
else
  echo 'STATUS: STOPPED or not started'
fi

echo
echo 'Active resumable files:'
find "${STAGE}" -type f -name '*.aria2' -print 2>/dev/null | while read -r control; do
  payload=${control%.aria2}
  logical=$(stat -f '%z' "${payload}" 2>/dev/null || echo 0)
  allocated=$(du -h "${payload}" 2>/dev/null | awk '{print $1}')
  printf '  %s | logical=%s bytes | disk=%s\n' "${payload}" "${logical}" "${allocated:-0}"
done

echo
df -h /private/tmp | tail -n 1

echo
echo 'Recent background log:'
tail -n 30 "${LOG_FILE}" 2>/dev/null || echo '  no log yet'
