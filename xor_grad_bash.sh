#!/bin/bash

rm xor_grad_progress.txt

for BOUNDS_SH in 0.5 1 5; do
    for MACRO_SH in True False; do
        for ACTIVATION_SH in sigmoid tanh relu; do
            echo "Bounds: " ${BOUNDS_SH}  ", Macro: " ${MACRO_SH} ", Activation: " ${ACTIVATION_SH}
            echo "Bounds: " ${BOUNDS_SH}  ", Macro: " ${MACRO_SH} ", Activation: " ${ACTIVATION_SH} >> xor_grad_progress.txt
            sed  "s/BOUNDS_SH/${BOUNDS_SH}/g" xor_fla_grad_bashable.py > xor_fla_grad_bashable_tmp1.py
            sed  "s/MACRO_SH/${MACRO_SH}/g" xor_fla_grad_bashable_tmp1.py > xor_fla_grad_bashable_tmp2.py
            sed  "s/ACTIVATION_SH/${ACTIVATION_SH}/g" xor_fla_grad_bashable_tmp2.py > xor_fla_grad_bashable_tmp3.py
            python xor_fla_grad_bashable_tmp3.py
        done
    done
done

for TMP in 1 2 3; do
    rm xor_fla_grad_bashable_tmp${TMP}.py
done