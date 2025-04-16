emt_yamls=(/Users/adrianshestakov/Work/stringScratch/rubin_sim_yamls/26*emt*.yaml)

for f in "${emt_yamls[@]}"; do
    echo $f;
    caffeinate python3 scripts/rubin_sim/calculate_emt.py \
    --config $f
done
