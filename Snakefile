# Partition panel. One rule, serial, twelve repetitions.

rule partition:
    output:
        "benchmarks/work/panel.txt",
    shell:
        "scripts/run_partition_panel.sh {output}"
