#!/usr/bin/perl -w

use strict;

if ($#ARGV!=2) {
  print "Usage: align.pl <first_pdb> <second_pdb> <output_file>\n";
  exit;
}

my $first_pdb = $ARGV[0];
my $second_pdb = $ARGV[1];
my $output_file = $ARGV[2];

$first_pdb =~ s/(\(|\))/\\$1/sg;
print $first_pdb;
`cp $first_pdb /tmp/first.pdb`;
`sed -i 's/HETATM/ATOM/g' /tmp/first.pdb`;
`cp $second_pdb /tmp/second.pdb`;
`sed -i 's/HETATM/ATOM/g' /tmp/second.pdb`;

if (-e $second_pdb)
{
	`/cs/staff/dina/utils/match.linux /tmp/first.pdb /tmp/second.pdb | head -n2 > $output_file`;
	`echo $first_pdb >> $output_file`;
	`echo $second_pdb >> $output_file`;
}
else
{
	print "Skipping $second_pdb\n"
}
