# Partition panel. Inputs are the sources the panel compiles and runs.

rule partition:
    input:
        "src/partition.rs",
        "src/lib.rs",
        "src/ptscotch_part.c",
        "src/scotch_part.c",
        "examples/partition_panel.rs",
        "scripts/run_partition_panel.sh",
        "target/release/examples/partition_panel",
        "benchmarks/work/ptscotch_part",
        "benchmarks/work/zoltan_rcb",
    output:
        "benchmarks/work/panel.txt",
    shell:
        "scripts/run_partition_panel.sh {output}"

rule asv_partition:
    input:
        "benchmarks/asv/partition.py",
        "src/partition.rs",
    output:
        "benchmarks/work/asv.txt",
    shell:
        ".venv/bin/asv run -b Partition --python=same --config benchmarks/asv.conf.json > {output}"
