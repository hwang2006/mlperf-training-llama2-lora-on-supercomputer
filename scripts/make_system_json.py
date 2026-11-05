#!/usr/bin/env python3
"""Write MLPerf system descriptions for Hangang to submission/<SUBMITTER>/systems/.
Usage: python3 scripts/make_system_json.py [SUBMITTER] [NODES ...]   (defaults: $SUBMITTER, 1 2 4 8 16 32 64)"""
import json, os, sys

sub = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SUBMITTER", "KISTI")
nodes = [int(x) for x in sys.argv[2:]] or [1, 2, 4, 8, 16, 32, 64]
root = os.environ.get("SUBMISSION_DIR", os.path.join(os.path.dirname(__file__), "..", "submission"))
out = os.path.join(root, sub, "systems"); os.makedirs(out, exist_ok=True)
for n in nodes:
    d = {
        "submitter": sub, "division": "closed", "status": "Available on-premise",
        "system_name": f"Hangang (HPE Cray EX254n, {4*n}x GH200-96GB) n{n}",
        "number_of_nodes": str(n),
        "host_processors_per_node": "4",
        "host_processor_model_name": "NVIDIA Grace (Arm Neoverse-V2)",
        "host_processor_core_count": "72",
        "host_processor_vcpu_count": "N/A",
        "host_processor_frequency": "3.4 GHz (max)",
        "host_processor_caches": "L3: 114 MiB per processor (456 MiB per node)",
        "host_processor_interconnect": "NVLink-C2C (CPU-GPU), Grace-to-Grace coherent fabric",
        "host_memory_capacity": "480 GB",
        "host_memory_configuration": "4x 120 GB LPDDR5X",
        "host_storage_type": "Lustre (shared parallel file system)",
        "host_storage_capacity": "107 PB",
        "host_networking": "HPE Slingshot 11, 4x Cassini NIC per node at 200 Gb/s (PCIe Gen5 x16)",
        "host_networking_topology": "Dragonfly (104 nodes per group)",
        "accelerators_per_node": "4",
        "accelerator_model_name": "NVIDIA GH200 Grace Hopper Superchip (H100 GPU)",
        "accelerator_host_interconnect": "NVLink-C2C",
        "accelerator_frequency": "1.98 GHz (max SM clock)",
        "accelerator_on-chip_memories": "N/A",
        "accelerator_memory_configuration": "HBM3",
        "accelerator_memory_capacity": "96 GB",
        "accelerator_interconnect": "NVLink (NV6 all-to-all within node); Slingshot 11 across nodes",
        "accelerator_interconnect_topology": "All-to-all within node; Dragonfly across nodes",
        "cooling": "Direct liquid-cooled (HPE Cray EX)",
        "hw_notes": "GPU power limit 900 W per GH200 superchip; default GPU vboost (not set)",
        "framework": "PyTorch 2.7.0a0 (NGC 25.04), NeMo 25.04-alpha.rc1, Megatron-core 0.12.0rc0",
        "other_software_stack": {
            "cuda_version": "12.9 (container, forward compatibility on driver 570.124.06)",
            "gpu_driver_version": "570.124.06",
            "nccl_version": "2.26.3",
            "aws_ofi_nccl_version": "1.14.0",
            "libfabric_version": "2.3.1 (HPE Cray, CXI provider)",
            "transformer_engine_version": "2.2.0",
            "torch_version": "2.7.0a0+79aa17489c.nv25.4",
            "slurm_version": "26.05.1 (Pyxis/Enroot)",
        },
        "operating_system": "SUSE Linux Enterprise Server 15 SP6 (kernel 6.4.0-150600.23.50-64kb)",
        "sw_notes": "Based on NVIDIA MLPerf Training v5.0 code (nyx_n1_ngc25.04_nemo) and Krai v6.0 train.py; "
                    "NCCL over Slingshot via host HPE Cray libfabric",
    }
    p = os.path.join(out, f"Hangang-GH200_n{n}.json")
    json.dump(d, open(p, "w"), indent=4); print("wrote", p)
