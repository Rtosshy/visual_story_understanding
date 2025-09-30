#!/bin/bash
qsub -I -P gch51711 -q rt_HG -l select=1 -l walltime=12:00:00