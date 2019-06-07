!!! note: need to use bedtools v. 2.25 -- groupby is broken in 2.26
load modules:
module load centos6
module load bedtools2/2.25.0-fasrc01

FIRST, GET FIMO OUTPUT INTO GENOMIC COORDINATES:
awk '{if (NR > 1) {{OFS="\t"} {match($2, /::([A-Za-z0-9\.\_]*)*:/, a); match($2, /:([0-9]*)\-/, b); match($2, /\-([0-9]*)\(/, c); match($2, /\(([+-])*\)$/, d)} if (d[1] == "+") {print a[1], b[1]+$3-1, b[1]+$4, $2, 0, $5, $1, $6, $8} else {print a[1], c[1]-$4, c[1]-$3+1, $2, 0, $5, $1, $6, $8}}}' ../02__mapped_motifs/hg19_evo_fimo_out/fimo.txt > hg19_evo_fimo.genomic_coords.txt

THEN, LIMIT TO SIGNIFICANT MOTIFS ONLY:
awk '{{OFS="\t"} if ($8 < 0.05) {print}}' hg19_evo_fimo.genomic_coords.txt > hg19_evo_fimo.sig.genomic_coords.txt

FIND # MOTIFS AND # BP COVERED:
awk '{{OFS="\t"} {print $8, $6, $1, $2, $3, $7}}' hg19_evo_fimo.sig.genomic_coords.txt | sort | uniq -f 2 | awk '{{OFS="\t"} {print $3, $4, $5, $6, $1, $2}}' | coverageBed -a ../../../data/01__design/00__mpra_list/hg19_evo.bed -b stdin > hg19_evo_fimo_sig.bp_covered.txt

FIND MAXIMUM COVERAGE:
bedtools intersect -wo -a ../../../data/01__design/00__mpra_list/hg19_evo.single_bp.bed -b hg19_evo_fimo.sig.genomic_coords.txt | awk '{{OFS="\t"} {print $4, $6, $14, $12, $1, $2, $3, $4, $16, $6, $7, $8, $9, $13}}' | sort | uniq -f 4 | awk '{{OFS="\t"} {print $5, $6, $7, $8, $9, $10, $11, $12, $13, $14}}' | sort | uniq | bedtools groupby -g 1,2,3,4 -c 10 -o count_distinct -i stdin | awk '{{OFS="\t"} {print $5, $4}}' | sort -k2,2 -k1,1nr | uniq -f 1 | awk '{{OFS="\t"} {print $2, $1}}' > hg19_evo_fimo_sig.max_coverage.txt


note that max coverage requires a bed file of every nucleotide in each 114p bp region (made using coverageBed -d -a promoters.bed -b promoters.bed!)
note that we need a bed file of the promoters used to find motifs and the motifs themselves in 12 column format (from 00__fimo_outputs)
also note that these files *de-dupe* by +/- (so if a motif shows up in exact same place with both + and -, only count once)


*** for the mosbat cluster output ***
bp covered:
coverageBed -a ../../data/00__index/0__all_tss/All.TSS.114bp.uniq.bed -b all_fimo_map.mosbat_cluster_info.merged_with_distinct_count.txt > all_fimo_map.mosbat_clusters.bp_covered.txt

max cov:
intersectBed -wo -a ../../data/00__index/0__all_tss/All.TSS.114bp.uniq.bed -b all_fimo_map.mosbat_cluster_info.merged_with_distinct_count.txt | bedtools groupby -g 4 -c 10 -o max -i stdin > all_fimo_map.mosbat_clusters.max_coverage.txt

where the merged_with_distinct_count file was merged based on the cluster number assigned from the heatmap
