cd /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos

python filter_dnsmos_tsv.py \
  --in_tsv /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/dns_from_vad_all.json_order.tsv \
  --out_tsv /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/dns_filtered_emilia.tsv \
  --min_dur 3.0 --max_dur 30.0 \
  --min_mos_ovr 3.0 --min_mos_sig 3.0 --min_mos_bak 2.5