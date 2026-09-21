#!/bin/zsh
# This watcher deliberately keeps mpirun as a child of this interactive PTY.
# It is not detached: every exit path waits for the launcher and its ranks.

set -u

guard_directory=/private/tmp/aspect_adjoint_mpi_memory_guard
mkdir -p "$guard_directory"

mpirun -n 2 build-debug/aspect \
  benchmarks/adjoint_time_dependent_temperature/adjoint_mpi_memory_diagnostic.prm \
  > "$guard_directory/aspect.log" 2>&1 &
launcher_pid=$!
captured=0
terminated=0

cleanup()
{
  if kill -0 "$launcher_pid" 2>/dev/null; then
    kill -TERM "$launcher_pid" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

while kill -0 "$launcher_pid" 2>/dev/null; do
  rank_pids=(${(f)"$(pgrep -f 'build-debug/aspect.*adjoint_mpi_memory_diagnostic.prm' || true)"})
  if (( ${#rank_pids} > 0 )); then
    process_table=$(/bin/ps -o pid= -o rss= -p ${(j:,:)rank_pids})
    total_rss_kb=$(print -r -- "$process_table" | awk '{total += $2} END {print total + 0}')
    print -r -- "$process_table" | awk \
      -v total="$total_rss_kb" \
      -v now="$(date +%H:%M:%S)" \
      '{print "MPI_RSS_SAMPLE time=" now " pid=" $1 " rss_kb=" $2} END {print "MPI_RSS_TOTAL time=" now " rss_kb=" total}'

    if (( total_rss_kb >= 512 * 1024 && captured == 0 )); then
      print "MPI_RSS_CAPTURE total_rss_kb=$total_rss_kb directory=$guard_directory"
      for pid in $rank_pids; do
        if (( $+commands[vmmap] )); then
          vmmap "$pid" > "$guard_directory/vmmap_${pid}.txt" 2>&1 || true
        fi
        if (( $+commands[sample] )); then
          sample "$pid" 3 > "$guard_directory/sample_${pid}.txt" 2>&1 || true
        fi
      done
      captured=1
    fi

    if (( total_rss_kb >= 1024 * 1024 )); then
      print "MPI_RSS_GUARD terminating MPI job at total_rss_kb=$total_rss_kb"
      for pid in $rank_pids; do
        kill -TERM "$pid" 2>/dev/null || true
      done
      cleanup
      terminated=1
      break
    fi
  fi
  sleep 1
done

wait "$launcher_pid"
wait_status=$?
print "MPI_RSS_GUARD_COMPLETE status=$wait_status terminated=$terminated log=$guard_directory/aspect.log"
exit "$wait_status"
