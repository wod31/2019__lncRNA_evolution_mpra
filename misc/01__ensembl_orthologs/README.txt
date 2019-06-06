downloaded using guidelines here: http://www.ensembl.info/2009/01/21/how-to-get-all-the-orthologous-genes-between-two-species/

ensembl 96
GRCh38.p12 human assembly



to get the ID map:
awk -F "\t" '{{OFS="\t"} if (NR > 5 && $3 == "gene") {match($9, /gene_id \"([A-Za-z0-9\.\_]*)\"/, a); match($9, /gene_type \"([A-Za-z0-9\.\_]*)\"/, b); match($9, /gene_name \"([A-Za-z0-9\.\_\-]*)\"/, c); print a[1], b[1], c[1]}}' /n/rinn_data2/users/kaia/annotation/mouse/gencode/gencode.vM13.annotation.gtf > gencode.vM13.GENE_ID_TO_NAME_AND_BIOTYPE_MAP.txt
