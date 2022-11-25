#!/usr/bin/perl -w

use strict;

if ($#ARGV!=2) {
  print "Usage: align.pl <first_pdb> <second_pdb> <output_file>\n";
  exit;
}

my $first_pdb = $ARGV[0];
my $second_pdb = $ARGV[1];
my $output_file = $ARGV[2];

`cp $first_pdb /tmp/first.pdb`;
`cp $second_pdb /tmp/second.pdb`;

`/cs/staff/dina/utils/match.linux /tmp/first.pdb /tmp/second.pdb | head -n2 > $output_file`;
`echo $first_pdb >> $output_file`;
`echo $second_pdb >> $output_file`;
