#!/bin/bash

a=$1
b=$2

cat "${a}.train" "${b}.train" | shuf > "${a}_${b}.train"
cat "${a}.dev" "${b}.dev" | shuf > "${a}_${b}.dev"

cp "${a}.test" "${a}_${b}.test"
cp "${a}.gen" "${a}_${b}.gen"
