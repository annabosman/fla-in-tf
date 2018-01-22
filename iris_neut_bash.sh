#!/bin/bash

rm iris_neut_progress.txt

for BOUNDS_SH in 0.5 1 5; do
    for MACRO_SH in True False; do
        for ACTIVATION_SH in sigmoid tanh relu; do
            echo "Bounds: " ${BOUNDS_SH}  ", Macro: " ${MACRO_SH} ", Activation: " ${ACTIVATION_SH}
            echo "Bounds: " ${BOUNDS_SH}  ", Macro: " ${MACRO_SH} ", Activation: " ${ACTIVATION_SH} >> iris_neut_progress.txt
            sed  "s/BOUNDS_SH/${BOUNDS_SH}/g" iris_neut_bash.py > iris_fla_neutral_bashable_tmp1.py
            sed  "s/MACRO_SH/${MACRO_SH}/g" iris_fla_neutral_bashable_tmp1.py > iris_fla_neutral_bashable_tmp2.py
            sed  "s/ACTIVATION_SH/${ACTIVATION_SH}/g" iris_fla_neutral_bashable_tmp2.py > iris_fla_neutral_bashable_tmp3.py
            python iris_fla_neutral_bashable_tmp3.py
        done
    done
done

for TMP in 1 2 3; do
    rm iris_fla_neutral_bashable_tmp${TMP}.py
done