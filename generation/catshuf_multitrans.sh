#!/bin/bash

a=$1
b=$2
c=$3

cat "${a}.train" "${b}.train" "${c}.train" | shuf > "${a}_multitrans.train"
cat "${a}.dev" "${b}.dev" "${c}.dev" | shuf > "${a}_multitrans.dev"

cp "${a}.test" "${a}_multitrans.test"
cp "${a}.gen" "${a}_multitrans.gen"
