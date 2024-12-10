data_path='../uci'
out_path='full_models'

mkdir ${out_path}


for data_i in "$data_path"/* 
do
  IFS="/" read -ra arr <<< "$data_i"
  name_i="${arr[-1]}"
  out_i="${out_path}/${name_i}"
  echo "$out_i"
  python3 train.py --data ${data_i} --model ${out_i}
done