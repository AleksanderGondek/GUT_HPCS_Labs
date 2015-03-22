 #!/bin/bash

echo "Compiling all files"
mpicc ./count_pi.c -lm -o ./Runnable/count_pi
mpicc ./count_with_ranges_pi.c -lm -o ./Runnable/count_with_ranges_pi
mpicc ./rational_count_pi.c -lm -o ./Runnable/rational_count_pi
mpicc ./rational_count_with_ranges_pi.c -lm -o ./Runnable/rational_count_with_ranges_pi

