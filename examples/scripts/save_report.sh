#!/bin/sh
# Run by the taskmaestro.command task. Upstream data is in $TASKMAESTRO_INPUT
# (JSON); the path of the saved report is returned through $TASKMAESTRO_OUTPUT.
set -eu
mkdir -p "$REPORT_DIR"
report="$REPORT_DIR/word_count.json"
cp "$TASKMAESTRO_INPUT" "$report"
# printf does not escape JSON: this output is valid only while REPORT_DIR has no
# '"' or '\'. Use a Python script (json.dump) for output built from arbitrary values.
printf '{"report": "%s"}' "$report" > "$TASKMAESTRO_OUTPUT"
echo "Saved $report"
