#!/usr/bin/perl -w

use strict;

if ($#ARGV!=1) {
	print "Usage: align_from_stats.pl <stats_file> <output_pdb>\n";
	exit;
}

my $stats_file = $ARGV[0];
my $output_pdb = $ARGV[1];

my $grep_line = `sed -n '1p' < $stats_file`;
my @tmp = split(' ', $grep_line);
my $first_pdb = `sed -n '2p' < $stats_file`;
my $second_pdb = `sed -n '3p' < $stats_file`;
print $second_pdb;
`(/cs/staff/dina/utils/pdb_trans $tmp[0] $tmp[1] $tmp[2] $tmp[3] $tmp[4] $tmp[5] < $second_pdb) > $output_pdb`;
# my $retina_molecule = `((grep RET $first_pdb) | grep HETATM) | grep '.'`;
# `echo "$retina_molecule" >> $output_pdb`;
