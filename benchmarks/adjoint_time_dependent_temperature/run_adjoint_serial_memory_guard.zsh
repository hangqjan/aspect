#!/bin/zsh
# Keep the serial validation process in this interactive PTY and enforce the
# same 500 MB capture / 1 GB stop thresholds as the MPI validation.

set -u

guard_directory=/private/tmp/aspect_adjoint_serial_memory_guard
mkdir -p "$guard_directory"

build-debug/aspect \
  benchmarks/adjoint_time_dependent_temperature/adjoint_serial_validation_direct.prm \
  > "$guard_directory/aspect.log" 2>&1 &
aspect_pid=$!
captured=0
terminated=0

cleanup()
{
  if kill -0 "$aspect_pid" 2>/dev/null; then
    kill -TERM "$aspect_pid" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

while kill -0 "$aspect_pid" 2>/dev/null; do
  rss_kb=$(/bin/ps -o rss= -p "$aspect_pid" | awk '{print $1 + 0}')
  print "SERIAL_RSS_SAMPLE time=$(date +%H:%M:%S) pid=$aspect_pid rss_kb=$rss_kb"

  if (( rss_kb >= 512 * 1024 && captured == 0 )); then
    print "SERIAL_RSS_CAPTURE rss_kb=$rss_kb directory=$guard_directory"
    if (( $+commands[vmmap] )); then
      vmmap "$aspect_pid" > "$guard_directory/vmmap_${aspect_pid}.txt" 2>&1 || true
    fi
    if (( $+commands[sample] )); then
      sample "$aspect_pid" 3 > "$guard_directory/sample_${aspect_pid}.txt" 2>&1 || true
    fi
    captured=1
  fi

  if (( rss_kb >= 1024 * 1024 )); then
    print "SERIAL_RSS_GUARD terminating job at rss_kb=$rss_kb"
    cleanup
    terminated=1
    break
  fi
  sleep 1
done

wait "$aspect_pid"
wait_status=$?
print "SERIAL_RSS_GUARD_COMPLETE status=$wait_status terminated=$terminated log=$guard_directory/aspect.log"
exit "$wait_status"
