data_path="binary_exp/data"
out_path="binary_exp/exp"

mkdir ${out_path}
#step=20

for data_i in "$data_path"/* 
do
  IFS="/" read -ra arr <<< "$data_i"
  name_i="${arr[-1]}"
  out_i="${out_path}/${name_i}"
  echo "$out_i"
  for j in {0..4}
  do
    let start=j*20
    python3 train.py --data ${data_i} --out_path ${out_i} --start ${start} --step "20"
  done
#  python3 train.py --data ${data_i} --model ${out_i}
done