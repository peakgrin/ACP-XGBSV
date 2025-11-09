# merge protTrans features
python feat_merge_col.py merged_prot.csv prot_xlnet.csv prot_bert_bfd.csv prot_bert.csv prot_albert.csv

# feature reduction
num_feat_1=1200
data_1=AAINDEX_ok.csv
data_fs_1=AAINDEX_ok_featSelect.csv
python feat_selection.py $num_feat_1 $data_1 $data_fs_1 1

num_feat_2=1200
data_2=merged_prot.csv
data_fs_2=merged_prot_featSelect.csv
python feat_selection.py $num_feat_2 $data_2 $data_fs_2 1

vote="XGBClassifier $data_fs_1 XGBClassifier BLOSUM62_ok.csv XGBClassifier BIT21_ok.csv XGBClassifier BIT20_ok.csv XGBClassifier 188D_ok.csv XGBClassifier CKSAAGP_ok.csv XGBClassifier $data_fs_2 XGBClassifier DDE_ok.csv"

#independent test
python classifierAny_metrics_sk.py 3 3 $vote

#cross-validation
python classifierAny_metrics_sk.py 4 3 $vote
