#!/bin/bash

export LC_NUMERIC=C

if [ ! -f "pi_calculation" ] || [ ! -f "matrix_vector" ]; then
    echo "ERROR: Executable files not found!"
    echo "Please compile first:"
    echo "  mpicc -O3 -o pi_calculation pi_calculation_mpi.c -lm"
    echo "  mpicc -O3 -o matrix_vector matrix_vector_multiply.c -lm"
    exit 1
fi

echo "Executable files found: pi_calculation, matrix_vector"
echo ""

rm -f pi_times.txt matrix_times.txt

echo "=== PART 1: PI Calculation ==="
declare -A base_times

for points in 100000000 500000000 1000000000; do
    echo ">> Base time for $points points (1 process)"
    mpiexec -n 1 ./pi_calculation $points 0
    echo ""
done

for processes in 2 3 4; do
    echo ">> Processes: $processes"
    
    for points in 100000000 500000000 1000000000; do
        echo "   Points: $points"
        
        for dist_type in 0 1; do
            echo "   Distribution: $dist_type"
            mpiexec -n $processes ./pi_calculation $points $dist_type
            echo "   ---"
        done
    done
    echo ""
done

echo "=== PART 2: Matrix-Vector Multiplication ==="
for processes in 1 2 3 4; do
    echo ">> Processes: $processes"
    
    if [ $processes -eq 1 ]; then
        for size in 1000 2000 3000; do
            echo "   Size: $size, Method: rows"
            mpiexec -n 1 ./matrix_vector rows $size 10
            echo "   ---"
        done
    else
        for method in "rows" "columns" "blocks"; do
            for size in 1000 2000 3000; do
                echo "   Size: $size, Method: $method"
                mpiexec -n $processes ./matrix_vector $method $size 10
                echo "   ---"
            done
        done
    fi
    echo ""
done

echo "================================================"
echo "TESTING COMPLETED!"
echo ""

calculate_speedup() {
    local base_time=$1
    local current_time=$2
    if (( $(echo "$current_time > 0" | bc -l) )); then
        speedup=$(echo "scale=4; $base_time / $current_time" | bc)
        printf "%.4f" "$speedup"
    else
        echo "0.0000"
    fi
}

format_time() {
    local time=$1
    echo "$time" | tr '.' ','
}

echo "=== PI CALCULATION RESULTS ==="
echo "Points      Procs Distribution Time        Speedup"
echo "----------- ----- ------------ ----------  ----------"

if [ -f "pi_times.txt" ]; then
    declare -A base_pi_times
    while read -r procs points dist time; do
        if [ "$procs" -eq 1 ] && [ "$dist" -eq 0 ]; then
            base_pi_times["$points"]=$time
        fi
    done < pi_times.txt
    
    while read -r procs points dist time; do
        base_time=${base_pi_times["$points"]}
        formatted_time=$(format_time "$time")
        
        if [ -n "$base_time" ]; then
            if [ "$procs" -eq 1 ] && [ "$dist" -eq 0 ]; then
                printf "%-11s %-5s %-12s %-11s %-10s\n" "$points" "$procs" "$dist" "$formatted_time" "1.0000"
            elif [ "$procs" -ne 1 ]; then
                speedup=$(calculate_speedup "$base_time" "$time")
                printf "%-11s %-5s %-12s %-11s %-10s\n" "$points" "$procs" "$dist" "$formatted_time" "$speedup"
            fi
        fi
    done < pi_times.txt | sort -k1,1n -k2,2n -k3,3n
else
    echo "PI results file not found"
fi

echo ""
echo "=== PI SUMMARY TABLE ==="
echo "| Points     | Sequential |    2 Processes    |    3 Processes    |    4 Processes    |"
echo "|            | Time       |   Time   | SpdUp  |   Time   | SpdUp  |   Time   | SpdUp  |"
echo "|------------|------------|----------|--------|----------|--------|----------|--------|"

if [ -f "pi_times.txt" ]; then
    for points in 100000000 500000000 1000000000; do
        time_1=""
        time_2=""
        time_3=""
        time_4=""
        
        while read -r procs points_val dist time; do
            if [ "$points_val" -eq "$points" ] && [ "$dist" -eq 0 ]; then
                case $procs in
                    1) time_1=$time ;;
                    2) time_2=$time ;;
                    3) time_3=$time ;;
                    4) time_4=$time ;;
                esac
            fi
        done < pi_times.txt
        
        if [ -n "$time_1" ] && [ -n "$time_2" ] && [ -n "$time_3" ] && [ -n "$time_4" ]; then
            speedup_2=$(calculate_speedup "$time_1" "$time_2")
            speedup_3=$(calculate_speedup "$time_1" "$time_3")
            speedup_4=$(calculate_speedup "$time_1" "$time_4")
            
            time_1_fmt=$(format_time "$time_1")
            time_2_fmt=$(format_time "$time_2")
            time_3_fmt=$(format_time "$time_3")
            time_4_fmt=$(format_time "$time_4")
            
            printf "| %-10s | %-10s | %-5s | %-5s | %-5s | %-5s | %-5s | %-5s |\n" \
                   "$points" "$time_1_fmt" "$time_2_fmt" "$speedup_2" "$time_3_fmt" "$speedup_3" "$time_4_fmt" "$speedup_4"
        fi
    done
fi

echo ""
echo "=== MATRIX-VECTOR RESULTS ==="
echo "Method  Size Procs Time        Speedup"
echo "------  ---- ----- ----------  ----------"

if [ -f "matrix_times.txt" ]; then
    declare -A base_matrix_times
    while read -r method size procs time; do
        if [ "$procs" -eq 1 ] && [ "$method" = "rows" ]; then
            base_matrix_times["$size"]=$time
        fi
    done < matrix_times.txt
    
    while read -r method size procs time; do
        base_time=${base_matrix_times["$size"]}
        formatted_time=$(format_time "$time")
        
        if [ -n "$base_time" ]; then
            if [ "$procs" -eq 1 ] && [ "$method" = "rows" ]; then
                printf "%-6s %-4s %-5s %-11s %-10s\n" "$method" "$size" "$procs" "$formatted_time" "1.0000"
            elif [ "$procs" -ne 1 ]; then
                speedup=$(calculate_speedup "$base_time" "$time")
                printf "%-6s %-4s %-5s %-11s %-10s\n" "$method" "$size" "$procs" "$formatted_time" "$speedup"
            fi
        fi
    done < matrix_times.txt | sort -k2,2n -k3,3n -k1,1
else
    echo "Matrix results file not found"
fi

echo ""
echo "=== MATRIX-VECTOR SUMMARY TABLE ==="
echo "| Size | Sequential |     2 Processes    |    3 Processes     |     4 Processes    |"
echo "|      | Time       |   Time   | Speedup |   Time   | Speedup |   Time   | Speedup |"
echo "|------|------------|----------|---------|----------|---------|----------|---------|"

if [ -f "matrix_times.txt" ]; then
    for size in 1000 2000 3000; do
        time_1=""
        time_2=""
        time_3=""
        time_4=""
        
        while read -r method size_val procs time; do
            if [ "$size_val" -eq "$size" ] && [ "$method" = "rows" ]; then
                case $procs in
                    1) time_1=$time ;;
                    2) time_2=$time ;;
                    3) time_3=$time ;;
                    4) time_4=$time ;;
                esac
            fi
        done < matrix_times.txt
        
        if [ -n "$time_1" ] && [ -n "$time_2" ] && [ -n "$time_3" ] && [ -n "$time_4" ]; then
            speedup_2=$(calculate_speedup "$time_1" "$time_2")
            speedup_3=$(calculate_speedup "$time_1" "$time_3")
            speedup_4=$(calculate_speedup "$time_1" "$time_4")
            
            time_1_fmt=$(format_time "$time_1")
            time_2_fmt=$(format_time "$time_2")
            time_3_fmt=$(format_time "$time_3")
            time_4_fmt=$(format_time "$time_4")
            
            printf "| %4d | %10s | %5s | %7s | %5s | %7s | %5s | %7s |\n" \
                   "$size" "$time_1_fmt" "$time_2_fmt" "$speedup_2" "$time_3_fmt" "$speedup_3" "$time_4_fmt" "$speedup_4"
        fi
    done
fi
