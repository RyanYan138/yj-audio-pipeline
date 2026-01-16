
cd /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos
python filter_dnsmos_tsv.py \
  --in_tsv /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/dns_from_vad_all.json_order.tsv \
  --out_tsv /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/dns_filtered_q70.tsv \
  --min_dur 2.0 --max_dur 20.0 \
  --keep_quantile 0.7
