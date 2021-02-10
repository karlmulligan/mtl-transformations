#!/bin/bash

for f in *.gen ; do
    name=$(basename -s .gen "${f}")

    head -5 "${name}.gen" > "${name}.gen_a5"
    head -50 "${name}.gen" > "${name}.gen_a50"
    head -500 "${name}.gen" > "${name}.gen_a500"

    tail -n 9995 "${name}.gen" > "${name}.gen_b5"
    tail -n 9950 "${name}.gen" > "${name}.gen_b50"
    tail -n 9500 "${name}.gen" > "${name}.gen_b500"

    cat "${name}.train" "${name}.gen_a5" | shuf > "nshot/${name}_5-shot.train"
    cp "${name}.dev" "nshot/${name}_5-shot.dev"
    cp "${name}.test" "nshot/${name}_5-shot.test"
    cp "${name}.gen_b5" "nshot/${name}_5-shot.gen"

    cat "${name}.train" "${name}.gen_a50" | shuf > "nshot/${name}_50-shot.train"
    cp "${name}.dev" "nshot/${name}_50-shot.dev"
    cp "${name}.test" "nshot/${name}_50-shot.test"
    cp "${name}.gen_b50" "nshot/${name}_50-shot.gen"

    cat "${name}.train" "${name}.gen_a500" | shuf > "nshot/${name}_500-shot.train"
    cp "${name}.dev" "nshot/${name}_500-shot.dev"
    cp "${name}.test" "nshot/${name}_500-shot.test"
    cp "${name}.gen_b500" "nshot/${name}_500-shot.gen"


    head -10 "${name}.gen" > "${name}.gen_a10"
    head -100 "${name}.gen" > "${name}.gen_a100"
    head -1000 "${name}.gen" > "${name}.gen_a1000"

    tail -n 9990 "${name}.gen" > "${name}.gen_b10"
    tail -n 9900 "${name}.gen" > "${name}.gen_b100"
    tail -n 9000 "${name}.gen" > "${name}.gen_b1000"

    cat "${name}.train" "${name}.gen_a10" | shuf > "nshot/${name}_10-shot.train"
    cp "${name}.dev" "nshot/${name}_10-shot.dev"
    cp "${name}.test" "nshot/${name}_10-shot.test"
    cp "${name}.gen_b10" "nshot/${name}_10-shot.gen"

    cat "${name}.train" "${name}.gen_a100" | shuf > "nshot/${name}_100-shot.train"
    cp "${name}.dev" "nshot/${name}_100-shot.dev"
    cp "${name}.test" "nshot/${name}_100-shot.test"
    cp "${name}.gen_b100" "nshot/${name}_100-shot.gen"

    cat "${name}.train" "${name}.gen_a1000" | shuf > "nshot/${name}_1000-shot.train"
    cp "${name}.dev" "nshot/${name}_1000-shot.dev"
    cp "${name}.test" "nshot/${name}_1000-shot.test"
    cp "${name}.gen_b1000" "nshot/${name}_1000-shot.gen"
done

rm *.gen_*
